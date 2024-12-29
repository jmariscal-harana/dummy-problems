import os
import torch
import pandas as pd
import numpy as np
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
from typing import Optional, Tuple, Any, Dict
from sklearn.model_selection import train_test_split
from collections import Counter


def count_target_frequency(dataloader) -> dict:
    targets = sorted([x.item() for xs in dataloader() for x in xs[1]])
    print(pd.DataFrame(targets).value_counts().to_string(header=False))

class SquarePad:
    def __init__(self, fill: Optional[float] = 0.0):
        self.fill = fill
        
    def __call__(self, image):
        h, w = image.shape[1:]
        max_wh = max(w, h)
        hp = (max_wh - w)
        vp = (max_wh - h)
        lp, rp = hp - (hp // 2), hp // 2  # left and right padding
        tp, bp = vp - (vp // 2), vp // 2  # top and bottom padding
        pad = (lp, rp, tp, bp)

        return torch.nn.functional.pad(image, pad, value=self.fill)


class LettersDataset(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.__load_data()

    def __load_data(self):
        if not self.root.exists():
            raise RuntimeError(f"Dataset not found at {self.root}.")

        self.image_dirs = sorted([d for d in self.root.iterdir()])
        labels = [label.stem[0] for label in self.image_dirs]
        self.labels_to_targets = {l: t for t, l in enumerate(sorted(set(labels)))}
        self.targets = torch.tensor([self.labels_to_targets[l] for l in labels])
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = decode_image(self.image_dirs[idx], mode=ImageReadMode.GRAY)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target
    
    def __len__(self) -> int:
        return len(self.image_dirs)


class PetsDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        indices: Optional[list] = None
    ):
        """
        Args:
            root (str): Root directory containing the image folders
            split_file (str): Path to the train/test split file
            transform (callable, optional): Transform to apply to images
            target_transform (callable, optional): Transform to apply to labels
        """
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )
        self.split_file = split_file
        self.indices = indices

        self.__load_data()

    def __load_data(self):
        if not self.root.exists():
            raise RuntimeError(f"Dataset not found at {self.root}.")

        # Load split file to get classes and filenames
        with open(self.split_file, 'r') as f:
            data = f.readlines()
        data = [d.replace("\n", "").split("\\") for d in data]  # remove newline and \
        data = pd.DataFrame(data).sort_values(0, ignore_index=True)  # sort by class
        if self.indices:
            data = data.iloc[self.indices]  # choose data based on indices

        self.labels = data[0].tolist()
        self.label_counts = Counter(self.labels)
        self.image_dirs = [self.root / d for d in data[0] + os.sep + data[1]]
        self.labels_to_targets = {l: t for t, l in enumerate(sorted(self.label_counts))}
        self.targets = torch.tensor([self.labels_to_targets[l] for l in self.labels])
        
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = decode_image(self.image_dirs[idx], mode=ImageReadMode.RGB)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.image_dirs)

    def get_class_weights(self) -> Dict[int, float]:
        """Calculate inverse frequency class weights."""
        
        return {key: 1/val for key, val in self.label_counts.items()}
    
    def get_sample_weights(self) -> np.ndarray:
        """Calculate sample weights for each class."""
        class_weights = self.get_class_weights()
        
        return np.array([class_weights[l] for l in self.labels])

class LettersDataModule(L.LightningDataModule):
    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.dataset_dir = settings['dataset_dir']
        self.num_workers = settings['num_workers']

        if settings["model_name"] == "tiny_vit_21m_224.dist_in22k_ft_in1k":
            self.transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224))])
        else:
            self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def prepare_data(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset not found at {self.dataset_dir}")

    def setup(self, stage: str):
        if stage == "fit":
            dataset = LettersDataset(self.dataset_dir / "train", transform=self.transform)
            self.train_dataset, self.val_dataset = train_test_split(
                dataset,
                test_size=0.2, 
                random_state=42,
                shuffle=True,
                stratify=dataset.targets)
        elif stage == "train":
            self.train_dataset = LettersDataset(self.dataset_dir / "train", transform=self.transform)
        elif stage == "test":
            self.test_dataset = LettersDataset(self.dataset_dir / "test", transform=self.transform)
        else:
            raise NotImplementedError(f"{stage} stage not implemented")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, num_workers=self.num_workers)

    def predict_dataloader(self):
        raise NotImplementedError("predict dataloader not implemented")


class PetsDataModule(L.LightningDataModule):
    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.dataset_dir = settings['dataset_dir']
        self.split_file_train = settings['dataset_dir'] / "train_set.txt"
        self.split_file_test = settings['dataset_dir'] / "test_set.txt"
        self.num_workers = settings['num_workers']
        self.batch_size = settings['batch_size']
        self.sampling = settings['sampling']

        self.train_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            SquarePad(),
            v2.Resize(settings['input_size']), 
            # v2.RandomResizedCrop(settings['input_size']),
            # v2.RandomHorizontalFlip(),
        ])

        self.test_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            SquarePad(),
            v2.Resize(settings['input_size']), 
        ])

    def get_train_val_indices(self):
        dataset = PetsDataset(
            root=self.dataset_dir,
            split_file=self.split_file_train,
            transform=None,
            )
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(len(dataset)), 
            [0.8, 0.2],
            generator=generator
        )

        return sorted(train_indices.indices), sorted(val_indices.indices)

    def prepare_data(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset not found at {self.dataset_dir}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train_indices, val_indices = self.get_train_val_indices()
            self.train_dataset = PetsDataset(
                root=self.dataset_dir,
                split_file=self.split_file_train,
                transform=self.train_transform,
                indices=train_indices
            )
            self.val_dataset = PetsDataset(
                root=self.dataset_dir,
                split_file=self.split_file_train,
                transform=self.test_transform,
                indices=val_indices
            )
        elif stage == "test":
            self.test_dataset = PetsDataset(
                root=self.dataset_dir,
                split_file=self.split_file_test,
                transform=self.test_transform
                )
        else:
            raise NotImplementedError(f"{stage} stage not implemented")
    
    def train_dataloader(self) -> DataLoader:
        if self.sampling == 'weighted':
            sample_weights = self.train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    settings = {
        "dataset_dir": Path("/home/ubuntu/data/pets"),
        "input_size": 224,
        "batch_size": 32,
        "sampling": "weighted",
        "num_workers": 7,
        "stage": "fit",
    }

    data = PetsDataModule(settings)
    data.setup("fit")
    data.setup("test")
    
    print("\nTarget frequencies in the training split:")
    count_target_frequency(data.train_dataloader)
    print("\nTarget frequencies in the validation split:")
    count_target_frequency(data.val_dataloader)
    print("\nTarget frequencies in the test split:")
    count_target_frequency(data.test_dataloader)