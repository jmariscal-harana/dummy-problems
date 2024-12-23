from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
import lightning as L
from typing import Tuple, Any
import torch
from torchvision.io import read_image

class LettersDataset(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.__get_image_dirs()

    def __get_image_dirs(self):
        if not self.root.exists():
            raise RuntimeError(f"Dataset not found at {self.root}.")

        self.image_dirs = sorted([f for f in self.root.iterdir()])
        
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = read_image(self.image_dirs[idx])
        target = self.image_dirs[idx].stem[0]

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

    def prepare_data(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset directory not found at {self.dataset_dir}")

    def setup(self, stage: str):
        if stage == "fit":
            dataset = LettersDataset(self.dataset_dir / "train")
            split = [0.8, 0.2]
            generator = torch.Generator().manual_seed(42)
            self.letters_train, self.letters_val = random_split(dataset, split, generator=generator)
        elif stage == "test":
            self.letters_test = LettersDataset(self.dataset_dir / "test")
        else:
            raise NotImplementedError(f"{stage} stage not implemented")

    def train_dataloader(self):
        return DataLoader(self.letters_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.letters_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.letters_test, batch_size=32)

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
    print("DataModule check successful")