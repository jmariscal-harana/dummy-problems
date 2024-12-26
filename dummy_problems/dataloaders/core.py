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

        if settings["model_name"] == "tiny_vit_21m_224.dist_in22k_ft_in1k":
            self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224))])
        else:
            self.transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def prepare_data(self):
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Dataset directory not found at {self.dataset_dir}")

    def setup(self, stage: str):
        if stage == "fit":
            dataset = LettersDataset(self.dataset_dir / "train", transform=self.transform)
            self.letters_train, self.letters_val = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True, stratify=dataset.targets)
        elif stage == "train":
            self.letters_train = LettersDataset(self.dataset_dir / "train", transform=self.transform)
        elif stage == "test":
            self.letters_test = LettersDataset(self.dataset_dir / "test", transform=self.transform)
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
        "num_workers": 15,
    }
    
    data = LettersDataModule(settings)
    data.setup("fit")
    data.setup("test")
    
    print("\nTarget frequencies in the training split:")
    count_target_frequency(data.train_dataloader)
    print("\nTarget frequencies in the validation split:")
    count_target_frequency(data.val_dataloader)
    print("\nTarget frequencies in the test split:")
    count_target_frequency(data.test_dataloader)