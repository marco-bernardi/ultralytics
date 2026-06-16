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
        from ultralytics.nn.tasks import OBBModel

        # nc is 17 for cards (4 suits + 13 ranks)
        model = OBBModel(cfg, nc=17, ch=self.data["channels"], verbose=verbose and RANK == -1)
        weights = weights or self.custom_weights
        if weights:
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


