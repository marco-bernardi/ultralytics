
import unittest
import numpy as np
from pathlib import Path
import shutil
import os
from ultralytics.data.dataset import CardsYOLODataset
from ultralytics.data.utils import check_det_dataset

class TestCardsYOLODataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_cards_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.test_dir / "images" / "train"
        self.labels_dir = self.test_dir / "labels" / "train"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy image
        self.img_file = self.images_dir / "test.jpg"
        import cv2
        cv2.imwrite(str(self.img_file), np.zeros((640, 640, 3), dtype=np.uint8))
        
        # Create a dummy label with 7 columns: suit_id rank_id x y w h angle
        # suit=0, rank=10, x=0.5, y=0.5, w=0.2, h=0.3, angle=0.5
        self.lbl_file = self.labels_dir / "test.txt"
        with open(self.lbl_file, "w") as f:
            f.write("0 10 0.5 0.5 0.2 0.3 0.5\n")
            
        self.data_yaml = {
            "path": str(self.test_dir.absolute()),
            "train": "images/train",
            "val": "images/train",
            "names": {i: f"class_{i}" for i in range(52)} # Dummy names
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cards_dataset_loading(self):
        dataset = CardsYOLODataset(
            img_path=str(self.images_dir),
            data=self.data_yaml,
            task="obb",
            augment=False
        )
        labels = dataset.get_labels()
        self.assertEqual(len(labels), 1)
        lb = labels[0]
        self.assertEqual(lb["cls"].shape, (1, 2))
        np.testing.assert_array_equal(lb["cls"][0], [0, 10])
        # Check if OBB is correctly parsed into 4 points (8 coords) in segments
        self.assertEqual(len(lb["segments"]), 1)
        self.assertEqual(lb["segments"][0].shape, (4, 2))

if __name__ == "__main__":
    unittest.main()
