import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from pycocotools.coco import COCO


class COCODataset:
    def __init__(self, dataset_path: str, split: str = "val2017"):
        self.dataset_path = Path(os.path.expanduser(dataset_path))
        self.split = split
        self.images_path = self.dataset_path / split
        self.coco = None
        self.image_ids = []

    def load_annotations(self, annotations_file: str = None) -> None:
        if annotations_file is None:
            annotations_file = (
                self.dataset_path / "annotations" / f"instances_{self.split}.json"
            )

        self.coco = COCO(str(annotations_file))
        self.image_ids = self.coco.getImgIds()

    def get_image_path(self, image_id: int) -> Path:
        image_info = self.coco.loadImgs(image_id)[0]
        return self.images_path / image_info["file_name"]

    def load_image(self, image_id: int) -> np.ndarray:
        image_path = self.get_image_path(image_id)
        image = cv2.imread(str(image_path))
        return image

    def load_batch(self, image_ids: List[int]) -> List[np.ndarray]:
        images = []
        for image_id in image_ids:
            image = self.load_image(image_id)
            if image is not None:
                images.append(image)
        return images

    def get_annotations(self, image_id: int) -> List[dict]:
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        return annotations

    def get_categories(self) -> List[dict]:
        return self.coco.loadCats(self.coco.getCatIds())

    def get_category_names(self) -> List[str]:
        categories = self.get_categories()
        return [cat["name"] for cat in categories]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[int, np.ndarray, List[dict]]:
        image_id = self.image_ids[index]
        image = self.load_image(image_id)
        annotations = self.get_annotations(image_id)
        return image_id, image, annotations


class COCOInferenceDataset:
    def __init__(self, dataset_path: str, split: str = "val2017"):
        self.dataset = COCODataset(dataset_path, split)
        self.dataset.load_annotations()

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for idx in range(len(self)):
            image_id, image, _ = self.dataset[idx]
            yield image_id, image

    def get_sample(self, index: int) -> Tuple[int, np.ndarray]:
        image_id, image, _ = self.dataset[index]
        return image_id, image
