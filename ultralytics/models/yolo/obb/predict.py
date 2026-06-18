# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    This predictor handles oriented bounding box detection tasks, processing images and returning results with rotated
    bounding boxes.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO OBB model.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb import OBBPredictor
        >>> args = dict(model="yolo26n-obb.pt", source=ASSETS)
        >>> predictor = OBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize OBBPredictor with optional model and data configuration overrides.

        Args:
            cfg (dict, optional): Default configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over the default config.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and rotation angles with shape (N, 7) where the
                last dimension contains [x, y, w, h, confidence, class_id, angle].
            img (torch.Tensor): The image after preprocessing with shape (B, C, H, W).
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and oriented bounding
                boxes.
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)


class CardsOBBPredictor(OBBPredictor):
    """Predictor for the multi-label CardsOBB head (4 suits + 13 ranks per box).

    Each ground-truth box carries two labels (a suit in 0-3 and a rank in 4-16). The standard OBBPredictor postprocess
    picks a single class via `cls.max(1)` and would discard the second label, so this predictor reimplements
    postprocess to:

    1. Split the 17 class scores into suit (0-3) and rank (4-16) groups and take the argmax of each (mutual exclusion
       within a group, matching `CardsRotatedTaskAlignedAssigner` and `CardsOBBLoss`).
    2. Combine them into a joint confidence `P(suit) * P(rank)` used for confidence filtering (matches the assigner
       alignment metric).
    3. Duplicate every surviving anchor into two detections — one carrying the suit id, one the rank id — so the output
       stays compatible with the single-class `Results.obb` object (same pattern as `CardsOBBValidator._prepare_batch`).
    4. Run rotated NMS per class (suit and rank are distinct class ids, so they do not suppress each other).

    Notes:
        - Do NOT enable `agnostic_nms=True`: suit and rank detections of the same box would suppress each other.
        - Tracking and `embed`/feature extraction are not supported (the parent OBBPredictor does not support them
          either).
        - `classes` filter works on the 0-16 ids, e.g. `classes=[0,1,2,3]` keeps only suits.

    Examples:
        >>> from ultralytics import CardsYOLO
        >>> model = CardsYOLO("best.pt")
        >>> results = model.predict("image.jpg", conf=0.25)
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process multi-label OBB predictions into a list of Results with duplicated suit/rank detections.

        Args:
            preds (torch.Tensor | tuple): Raw model predictions. Either BCN `(bs, 4+nc+1, N)` from the standard
                inference path (box xywh, 17 sigmoid class scores, 1 angle) or BNC `(bs, max_det, 4+nc+2)` from the
                end-to-end path (box xywh, 17 class scores, conf, angle).
            img (torch.Tensor): Preprocessed input image tensor.
            orig_imgs (torch.Tensor | list): Original images before preprocessing.
            **kwargs (Any): Additional arguments. `iou` overrides `self.args.iou`.

        Returns:
            (list[Results]): One Results per image, each with an `obb` attribute of shape (num_det, 7) where each row
                is `[x, y, w, h, conf, cls, angle]` and `cls` is a single id in 0-16 (suit or rank).
        """
        from ultralytics.utils.metrics import batch_probiou
        from ultralytics.utils.nms import TorchNMS

        pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        # Normalize to BNC (bs, N, channels). CardsOBB raw inference returns BCN (bs, 22, N); end-to-end returns
        # BNC (bs, max_det, 23) already. Distinguish by number of channels: 22 (raw) or 23 (end2end) on dim 1 => BCN.
        if pred.shape[1] in (22, 23):  # BCN (channels, anchors) -> transpose to (anchors, channels)
            pred = pred.transpose(-1, -2)
        # Layout (last column is always angle): raw = [xywh(4), cls(17), angle(1)] = 22 cols;
        # end2end = [xywh(4), cls(17), conf(1), angle(1)] = 23 cols. cls is always at [4:21].
        box = pred[..., :4]  # (bs, N, 4) xywh
        cls = pred[..., 4:21]  # (bs, N, 17) sigmoid scores
        angle = pred[..., -1:]  # (bs, N, 1)

        # Best suit and best rank per anchor (mutual exclusion within each group).
        suit_conf, suit_id = cls[..., 0:4].max(-1)  # (bs, N)
        rank_conf, rank_id = cls[..., 4:17].max(-1)  # (bs, N)
        joint_conf = suit_conf * rank_conf  # matches CardsRotatedTaskAlignedAssigner alignment metric

        conf_thres = self.args.conf
        iou_thres = kwargs.pop("iou", self.args.iou)
        max_det = self.args.max_det
        max_wh = 7680
        classes = self.args.classes
        if classes is not None:
            classes = torch.tensor(classes, device=pred.device)
        agnostic = self.args.agnostic_nms

        bs = pred.shape[0]
        output = []
        for i in range(bs):
            keep = joint_conf[i] > conf_thres
            if not keep.any():
                output.append(pred.new_zeros((0, 7)))
                continue
            b = box[i][keep]  # (K, 4)
            a = angle[i][keep]  # (K, 1)
            sc = joint_conf[i][keep]  # (K,)
            sid = suit_id[i][keep]  # (K,)
            rid = rank_id[i][keep]  # (K,)

            # Duplicate each anchor into two rows: suit (cls=sid) then rank (cls=rid+4).
            boxes2 = b.repeat_interleave(2, dim=0)  # (2K, 4)
            angle2 = a.repeat_interleave(2, dim=0)  # (2K, 1)
            conf2 = sc.repeat_interleave(2)  # (2K,)
            cls2 = torch.stack([sid, rid + 4], dim=1).flatten()  # (2K,)

            # Optional class filter (0-16 ids).
            if classes is not None:
                filt = (cls2[:, None] == classes[None, :]).any(1)
                boxes2, angle2, conf2, cls2 = boxes2[filt], angle2[filt], conf2[filt], cls2[filt]
            if boxes2.shape[0] == 0:
                output.append(pred.new_zeros((0, 7)))
                continue

            # Rotated NMS per class: offset xy by cls*max_wh so different classes do not suppress each other
            # (same scheme as non_max_suppression with rotated=True, agnostic=False).
            c = cls2 * (0 if agnostic else max_wh)
            boxes_nms = torch.cat([boxes2[:, :2] + c[:, None], boxes2[:, 2:4], angle2], dim=-1)  # xywhr
            idx = TorchNMS.fast_nms(boxes_nms, conf2, iou_thres, iou_func=batch_probiou)[:max_det]
            det = torch.cat([boxes2[idx], conf2[idx, None], cls2[idx, None].float(), angle2[idx]], dim=-1)  # (n, 7)
            output.append(det)

        if not isinstance(orig_imgs, list):  # torch.Tensor batch -> list of numpy HWC BGR
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(output, img, orig_imgs, **kwargs)
