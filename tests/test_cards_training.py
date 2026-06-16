
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



if __name__ == "__main__":
    unittest.main()
