# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for detecting
    objects at arbitrary angles rather than just axis-aligned rectangles.

    Attributes:
        loss_names (tuple): Names of the loss components used during training including box_loss, cls_loss, dfl_loss,
            and angle_loss.

    Methods:
        get_model: Return OBBModel initialized with specified config and weights.
        get_validator: Return an instance of OBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize an OBBTrainer object for training Oriented Bounding Box (OBB) models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and model
                configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here will
                take precedence over those in cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """Return OBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (OBBModel): Initialized OBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
        """
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


from ultralytics.models.yolo.obb.val import OBBValidator
import torch

class CardsOBBValidator(OBBValidator):
    """Custom validator to duplicate GT boxes for independent suit and rank mAP evaluation."""

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess multi-label OBB predictions with suit/rank split NMS.

        Overrides the standard OBBValidator postprocess (which uses non_max_suppression with
        multi_label=True on all 17 BCE channels) because that produces ~10-15 detections per
        anchor (many channels above the low val conf threshold), flooding the evaluation with
        false positives and capping precision/mAP. Instead, this takes the argmax of the suit
        group (0-3) and rank group (4-16) separately, emits at most 2 detections per anchor
        (one suit, one rank), and runs rotated NMS per class.

        Returns:
            (list[dict]): One dict per image with keys 'bboxes' (N,5 xywhr), 'conf' (N,),
                'cls' (N,) where cls is a single id in 0-16 (suit or rank).
        """
        from ultralytics.utils.metrics import batch_probiou
        from ultralytics.utils.nms import TorchNMS

        pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        # Normalize to BNC. CardsOBB raw: BCN (bs,22,N); end2end: BNC (bs,max_det,23).
        if pred.shape[1] in (22, 23):
            pred = pred.transpose(-1, -2)
        box = pred[..., :4]  # xywh
        cls = pred[..., 4:21]  # 17 sigmoid scores
        angle = pred[..., -1:]  # angle

        suit_conf, suit_id = cls[..., 0:4].max(-1)
        rank_conf, rank_id = cls[..., 4:17].max(-1)

        conf_thres = self.args.conf
        iou_thres = self.args.iou
        max_det = self.args.max_det
        max_wh = 7680

        bs = pred.shape[0]
        outputs = []
        for i in range(bs):
            # Keep anchors where either suit or rank exceeds threshold (independent filtering).
            s_keep = suit_conf[i] > conf_thres
            r_keep = rank_conf[i] > conf_thres
            if not (s_keep.any() or r_keep.any()):
                outputs.append({"bboxes": pred.new_zeros((0, 5)), "conf": pred.new_zeros((0,)), "cls": pred.new_zeros((0,))})
                continue

            # Suit detections
            box_s = box[i][s_keep]
            angle_s = angle[i][s_keep]
            conf_s = suit_conf[i][s_keep]
            cls_s = suit_id[i][s_keep]

            # Rank detections
            box_r = box[i][r_keep]
            angle_r = angle[i][r_keep]
            conf_r = rank_conf[i][r_keep]
            cls_r = rank_id[i][r_keep] + 4

            # Concat suit + rank rows
            boxes2 = torch.cat([box_s, box_r], 0)
            angle2 = torch.cat([angle_s, angle_r], 0)
            conf2 = torch.cat([conf_s, conf_r], 0)
            cls2 = torch.cat([cls_s, cls_r], 0)

            if boxes2.shape[0] == 0:
                outputs.append({"bboxes": pred.new_zeros((0, 5)), "conf": pred.new_zeros((0,)), "cls": pred.new_zeros((0,))})
                continue

            # Rotated NMS per class (offset xy by cls*max_wh so different classes don't suppress).
            c = cls2 * max_wh
            boxes_nms = torch.cat([boxes2[:, :2] + c[:, None], boxes2[:, 2:4], angle2], dim=-1)
            idx = TorchNMS.fast_nms(boxes_nms, conf2, iou_thres, iou_func=batch_probiou)[:max_det]

            bboxes = torch.cat([boxes2[idx], angle2[idx]], dim=-1)  # (n, 5) xywhr
            outputs.append({"bboxes": bboxes, "conf": conf2[idx], "cls": cls2[idx].float()})

        return outputs

    def _prepare_batch(self, si: int, batch: dict):
        """Prepare batch data for OBB validation by duplicating GT boxes for suit and rank."""
        # First call super to handle standard formatting and SCALING of bboxes!
        res = super()._prepare_batch(si, batch)
        
        # Check original 2-column classes directly from the batch using the si index
        idx = batch["batch_idx"] == si
        cls_original = batch["cls"][idx].long()  # (n_boxes, 2)
        
        # If there are boxes and they are in the 2-column format
        if cls_original.shape[0] > 0 and cls_original.shape[1] == 2:
            suit_cls = cls_original[:, 0]  # Suit ID (0-3)
            rank_cls = cls_original[:, 1] + 4  # Rank ID (4-16)
            
            # Create a 1D tensor [suit1, rank1, suit2, rank2, ...]
            new_cls = torch.stack([suit_cls, rank_cls], dim=1).flatten().float()
            
            # Duplicate the ALREADY SCALED bboxes from res
            new_bbox = res["bboxes"].repeat_interleave(2, dim=0)
            
            # Replace in result
            res["cls"] = new_cls
            res["bboxes"] = new_bbox
        else:
            # If standard 1D classes (e.g. empty image), res["cls"] is already correctly squeezed by super()
            res["cls"] = cls_original.squeeze(-1).float()
            
        return res


class CardsOBBTrainer(OBBTrainer):
    """
    A class extending the OBBTrainer class for training Playing Cards multi-label model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize CardsOBBTrainer and handle custom arguments."""
        if overrides is None:
            overrides = {}
        # Pop custom arguments before passing to super to avoid validation errors
        self.custom_weights = overrides.pop("weights", None)
        self.freeze_backbone_flag = overrides.pop("freeze_backbone", False)
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):

        """Build CardsYOLODataset for training or validation."""
        from ultralytics.data import CardsYOLODataset
        from ultralytics.utils import colorstr

        # Use stride from model if available, otherwise default to 32
        stride = 32
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "stride"):
                stride = int(self.model.stride.max())
        gs = max(stride, 32)

        return CardsYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel with CardsOBB head."""
        from ultralytics.nn.tasks import OBBModel, load_checkpoint

        # nc is 17 for cards (4 suits + 13 ranks)
        model = OBBModel(cfg, nc=17, ch=self.data["channels"], verbose=verbose and RANK == -1)
        weights = weights or self.custom_weights
        if weights:
            if isinstance(weights, (str, Path)):
                weights, _ = load_checkpoint(weights)
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set model attributes and freeze backbone if requested."""
        super().set_model_attributes()
        # Check if freeze_backbone was passed during initialization
        if self.freeze_backbone_flag:
            from ultralytics.utils import LOGGER

            LOGGER.info("Freezing backbone as requested (freeze_backbone=True)")
            # self.model.model is the Sequential containing all layers
            # The last layer [-1] is the CardsOBB head
            for param in self.model.model[:-1].parameters():
                param.requires_grad = False

    def get_validator(self):
        """Return an instance of CardsOBBValidator for validation of custom multi-label YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss"
        return CardsOBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


