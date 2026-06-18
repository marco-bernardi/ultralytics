
import unittest
import torch
import os
import shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import CardsOBBTrainer
from ultralytics.data.dataset import CardsYOLODataset
from ultralytics.nn.modules.head import CardsOBB
from ultralytics.utils.loss import CardsOBBLoss

class TestCardsTraining(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_cards_train")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        # Create dummy data
        (self.test_dir / "images" / "train").mkdir(parents=True)
        (self.test_dir / "labels" / "train").mkdir(parents=True)
        
        # Dummy image
        from PIL import Image
        img = Image.new('RGB', (640, 640), color=(73, 109, 137))
        img.save(self.test_dir / "images" / "train" / "test.jpg")
        
        # Dummy label: suit rank x y w h angle
        with open(self.test_dir / "labels" / "train" / "test.txt", "w") as f:
            f.write("0 5 0.5 0.5 0.2 0.3 0.1\n")
            
        # Dummy data.yaml
        with open(self.test_dir / "data.yaml", "w") as f:
            f.write(f"path: {self.test_dir.absolute()}\n")
            f.write("train: images/train\n")
            f.write("val: images/train\n")
            f.write("nc: 17\n")
            f.write("names: ['S', 'H', 'D', 'C', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']\n")

        # Create custom model YAML
        self.model_yaml = self.test_dir / "yolo26-obb-cards.yaml"
        # Read original yolo26-obb.yaml and modify it
        with open("ultralytics/cfg/models/26/yolo26-obb.yaml", "r") as f:
            lines = f.readlines()
        
        with open(self.model_yaml, "w") as f:
            for line in lines:
                if "nc: 80" in line:
                    f.write("nc: 17\n")
                elif "OBB26" in line:
                    f.write(line.replace("OBB26", "CardsOBB"))
                else:
                    f.write(line)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_trainer_initialization(self):
        # Initialize trainer without custom arguments that fail validation
        args = dict(model=str(self.model_yaml), data=str(self.test_dir / "data.yaml"), epochs=1, imgsz=640)
        trainer = CardsOBBTrainer(overrides=args)
        # Set custom attribute manually
        trainer.args.freeze_backbone = True
        
        # Verify model initialization
        model = trainer.get_model(cfg=args['model'])
        self.assertIsInstance(model.model[-1], CardsOBB)
        
        # Verify trainer sets attributes and freezes backbone
        trainer.model = model.to(trainer.device)
        trainer.set_model_attributes()
        
        # Check if backbone is frozen
        # layers 0 to -2 should be frozen
        for i, m in enumerate(trainer.model.model[:-1]):
            for p in m.parameters():
                self.assertFalse(p.requires_grad, f"Layer {i} should be frozen")
        
        # Check if head is NOT frozen
        for p in trainer.model.model[-1].parameters():
            self.assertTrue(p.requires_grad, "Head should NOT be frozen")

    def test_loss_selection(self):
        # Verify that CardsOBBTrainer uses CardsOBBLoss
        args = dict(model=str(self.model_yaml), data=str(self.test_dir / "data.yaml"), epochs=1, imgsz=640)
        trainer = CardsOBBTrainer(overrides=args)
        model = trainer.get_model(cfg=args['model'])
        trainer.model = model.to(trainer.device)
        
        # Attach args to model so init_criterion doesn't fail
        trainer.model.args = trainer.args
        
        from ultralytics.nn.tasks import OBBModel
        self.assertIsInstance(trainer.model, OBBModel)
        
        # OBBModel.criterion for obb task should be v8OBBLoss or CardsOBBLoss
        criterion = trainer.model.init_criterion()
        from ultralytics.utils.loss import E2ELoss
        if isinstance(criterion, E2ELoss):
            self.assertIsInstance(criterion.one2many, CardsOBBLoss)
        else:
            self.assertIsInstance(criterion, CardsOBBLoss)

    def test_predictor_postprocess(self):
        # Verify CardsOBBPredictor postprocess splits suit/rank and duplicates detections
        from ultralytics.models.yolo.obb.predict import CardsOBBPredictor
        from types import SimpleNamespace

        # Build a fake predictor with minimal args needed by postprocess
        predictor = CardsOBBPredictor.__new__(CardsOBBPredictor)
        predictor.args = SimpleNamespace(
            conf=0.5, iou=0.45, max_det=300, classes=None, agnostic_nms=False
        )
        predictor.model = SimpleNamespace(names={i: str(i) for i in range(17)})

        # Build fake preds: BCN (1, 22, 3) = [box(4), cls(17), angle(1)]
        # cls layout (after transpose to BNC, channels 4..20 = 17 sigmoid scores):
        #   suit channels: 4..7 (suit id 0..3)
        #   rank channels: 8..20 (rank rel id 0..12 -> cls id 4..16)
        # 3 anchors:
        #  anchor 0: suit=0 (0.9), rank rel=1 (0.8) -> joint 0.72 -> kept, dup -> cls 0 and 5
        #  anchor 1: suit=1 (0.6), rank rel=6 (0.6) -> joint 0.36 -> filtered out
        #  anchor 2: suit=2 (0.95), rank rel=12 (0.95) -> joint 0.9025 -> kept, dup -> cls 2 and 16
        preds = torch.zeros(1, 22, 3)
        # box xywh (distinct x per anchor to avoid NMS suppression between them)
        preds[0, 0:4, 0] = torch.tensor([10.0, 10.0, 50.0, 30.0])
        preds[0, 0:4, 1] = torch.tensor([200.0, 10.0, 50.0, 30.0])
        preds[0, 0:4, 2] = torch.tensor([400.0, 10.0, 50.0, 30.0])
        # cls: suit on channels 4..7, rank on channels 8..20
        preds[0, 4 + 0, 0] = 0.9    # anchor 0 suit id 0 -> cls 0
        preds[0, 8 + 1, 0] = 0.8    # anchor 0 rank rel 1 -> cls 5
        preds[0, 4 + 1, 1] = 0.6    # anchor 1 suit id 1
        preds[0, 8 + 6, 1] = 0.6    # anchor 1 rank rel 6 -> cls 10
        preds[0, 4 + 2, 2] = 0.95   # anchor 2 suit id 2 -> cls 2
        preds[0, 8 + 12, 2] = 0.95  # anchor 2 rank rel 12 -> cls 16
        # angle (channel 21)
        preds[0, 21, :] = 0.1

        # Fake img and orig_imgs
        img = torch.zeros(1, 3, 640, 640)
        import numpy as np
        orig_imgs = [np.zeros((640, 640, 3), dtype=np.uint8)]
        predictor.batch = [ ["fake.jpg"] ]

        results = predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 1)
        obb = results[0].obb
        # Two anchors kept (0 and 2), each duplicated into suit+rank -> 4 detections
        self.assertEqual(obb.data.shape[0], 4)
        cls_ids = sorted(obb.cls.tolist())
        # anchor 0 -> suit cls 0 and rank cls 5; anchor 2 -> suit cls 2 and rank cls 16
        self.assertEqual(cls_ids, [0, 2, 5, 16])

    def test_predictor_classes_filter(self):
        # Verify classes filter keeps only requested class ids (0-16)
        from ultralytics.models.yolo.obb.predict import CardsOBBPredictor
        from types import SimpleNamespace

        predictor = CardsOBBPredictor.__new__(CardsOBBPredictor)
        predictor.args = SimpleNamespace(
            conf=0.5, iou=0.45, max_det=300, classes=[0, 1, 2, 3], agnostic_nms=False
        )  # keep only suits
        predictor.model = SimpleNamespace(names={i: str(i) for i in range(17)})

        preds = torch.zeros(1, 22, 2)
        preds[0, 0:4, 0] = torch.tensor([10.0, 10.0, 50.0, 30.0])
        preds[0, 0:4, 1] = torch.tensor([200.0, 10.0, 50.0, 30.0])
        preds[0, 4 + 0, 0] = 0.9
        preds[0, 8 + 1, 0] = 0.8  # rank cls 5 -> filtered out by classes=[0,1,2,3]
        preds[0, 4 + 1, 1] = 0.9
        preds[0, 8 + 2, 1] = 0.8  # rank cls 6 -> filtered out
        preds[0, 21, :] = 0.1

        img = torch.zeros(1, 3, 640, 640)
        import numpy as np
        orig_imgs = [np.zeros((640, 640, 3), dtype=np.uint8)]
        predictor.batch = [["fake.jpg"]]

        results = predictor.postprocess(preds, img, orig_imgs)
        cls_ids = sorted(results[0].obb.cls.tolist())
        # Only suits (cls 0 and 1) survive
        self.assertEqual(cls_ids, [0, 1])

    def test_cards_yolo_task_map(self):
        # Verify CardsYOLO exposes Cards-specific trainer/validator/predictor for the obb task
        from ultralytics import CardsYOLO
        from ultralytics.models.yolo.obb import CardsOBBTrainer, CardsOBBValidator, CardsOBBPredictor

        # Instantiate without loading a real checkpoint: bypass __init__ to inspect task_map only
        instance = CardsYOLO.__new__(CardsYOLO)
        tm = CardsYOLO.task_map.fget(instance)
        self.assertIs(tm["obb"]["trainer"], CardsOBBTrainer)
        self.assertIs(tm["obb"]["validator"], CardsOBBValidator)
        self.assertIs(tm["obb"]["predictor"], CardsOBBPredictor)



if __name__ == "__main__":
    unittest.main()
