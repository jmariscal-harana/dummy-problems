from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import lightning as L
from typing import Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

DEFAULT_TRANSFORM = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

def count_target_frequency(dataloader) -> dict:
    targets = sorted([x.item() for xs in dataloader() for x in xs[1]])
    print(pd.DataFrame(targets).value_counts().to_string(header=False))


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
        img = decode_image(self.image_dirs[idx], mode=ImageReadMode.RGB)
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return len(self.image_dirs)


class LettersDataModule(L.LightningDataModule):
    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.dataset_dir = settings['dataset_dir']
        self.settings = settings

    def prepare_data(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset directory not found at {self.dataset_dir}")

    def setup(self, stage: str):
        if stage == "fit":
            dataset = LettersDataset(self.dataset_dir / "train", transform=DEFAULT_TRANSFORM)
            self.letters_train, self.letters_val = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True, stratify=dataset.targets)
        elif stage == "test":
            self.letters_test = LettersDataset(self.dataset_dir / "test", transform=DEFAULT_TRANSFORM)
        else:
            raise NotImplementedError(f"{stage} stage not implemented")

    def train_dataloader(self):
        return DataLoader(self.letters_train, batch_size=32, num_workers=self.settings['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.letters_val, batch_size=32, num_workers=self.settings['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.letters_test, batch_size=32, num_workers=self.settings['num_workers'])

    def predict_dataloader(self):
        raise NotImplementedError("predict dataloader not implemented")
    
if __name__ == "__main__":
    settings = {
        "dataset_dir": Path("/home/ubuntu/data/letters_dataset"),
        "stage": "fit",
    }
    
    data = LettersDataModule(settings)
    data.setup("fit")
    data.setup("test")
    
    print("Training split size:", len(data.train_dataloader()))
    print("Validation split size:", len(data.val_dataloader()))
    print("Test split size:", len(data.test_dataloader()))

    print("\nTarget frequencies in the training split:")
    count_target_frequency(data.train_dataloader)
    print("\nTarget frequencies in the validation split:")
    count_target_frequency(data.val_dataloader)
    print("\nTarget frequencies in the test split:")
    count_target_frequency(data.test_dataloader)