from pathlib import Path
from dummy_problems.dataloaders import LettersDataModule, PetsDataModule
import lightning as L
import torch
import torch.nn as nn
from timm import create_model
import torchmetrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pickle


class SupportVectorMachine:
    def __init__(self, settings):
        self.settings = settings

        # Create pipeline with scaling and SVM and set up a grid search
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('svm', SVC())
        ])

        param_grid = {
            'svm__C': [1, 10, 100],
            'svm__gamma': ['scale', 0.1, 0.01],
            }

        self.model = GridSearchCV(
            estimator,
            param_grid,
            cv=2,
            n_jobs=settings["num_workers"],
            verbose=2
        )

    def __preprocess_data(self, data):
        images = torch.stack([d[0][0] for d in data])
        images = images.reshape(images.shape[0], -1)
        targets = torch.stack([d[1] for d in data])

        return images, targets

    def save_model(self):
        with open(self.settings["checkpoint"], 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        if not Path(self.settings["checkpoint"]).exists():
            raise RuntimeError(f"Model not found at {self.settings['checkpoint']}.")
        with open(self.settings["checkpoint"], 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, data):
        images, targets = self.__preprocess_data(data)

        self.model.fit(images, targets)
        self.save_model()
        print(f"Best parameters: {self.model.best_params_}")

    def test(self, data):
        images, targets = self.__preprocess_data(data)
        labels = list(data.labels_to_targets.keys())
        
        self.load_model()
        predictions = self.model.predict(images)
        
        print(classification_report(targets, predictions, target_names=labels, digits=3))
        print(confusion_matrix(targets, predictions))


class ConvNet(nn.Module):
    def __init__(self, settings):
        super(ConvNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=settings["num_channels"],
            out_channels=32,
            kernel_size=3,
            padding='same'
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding='same'
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding='same'
        )
        self.relu3 = nn.ReLU()
        
        # Calculate the size of flattened features
        self.flatten_size = 64 * 32 * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, settings["num_classes"])
    
    def forward(self, x):
        # Convolutional blocks
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, self.flatten_size)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x


class DLClassificationModel(L.LightningModule):
    def __init__(self, settings):
        super().__init__()
        self.save_hyperparameters()
        self.settings = settings

        if settings["model_name"] == "ConvNet":
            self.model = ConvNet(settings)
        else:
            self.model = create_model(
                settings["model_name"],
                pretrained=True,
                num_classes=settings["num_classes"],
                in_chans=settings["num_channels"],
                )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=settings["num_classes"])
        self.accuracy_val = torchmetrics.classification.Accuracy(task="multiclass", num_classes=settings["num_classes"])
        self.accuracy_test = torchmetrics.classification.Accuracy(task="multiclass", num_classes=settings["num_classes"])

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        
        self.accuracy_train(outputs, targets)
        self.log('train_acc', self.accuracy_train, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        self.accuracy_val(outputs, targets)
        self.log('val_acc', self.accuracy_val, on_step=True, on_epoch=True)

    def test_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
                
        self.accuracy_test(outputs, targets)
        self.log('test_acc', self.accuracy_test, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())

        return optimizer


MODEL_TYPES = {
    "SVM": SupportVectorMachine,
    "DL": DLClassificationModel,
}

# Model names (for reference only)
MODEL_NAMES = {
    "SVM",
    "ConvNet",
    "tiny_vit_21m_224.dist_in22k_ft_in1k",
    }

if __name__ == "__main__":
    settings =  {
        "dataset_dir": Path("/home/ubuntu/data/pets"),
        "input_size": 224,
        "batch_size": 32,
        "sampling": "weighted",
        "num_workers": 7,

        "model_type": "DL",
        "model_name": "tiny_vit_21m_224.dist_in22k_ft_in1k",
        "num_classes": 3,
        "num_channels": 3,
        "stage": "fit",
        # "checkpoint": "lightning_logs/version_2/checkpoints/epoch=9-step=70.ckpt",
    }
    
    data = PetsDataModule(settings)
    model = MODEL_TYPES[settings['model_type']](settings)

    if settings['model_type'] == "SVM":
        data.setup("train")
        model.fit(data.train_dataset)
        data.setup("test")
        model.test(data.train_dataset)  # sanity check
        model.test(data.test_dataset)
    
    elif settings['model_type'] == "DL":
        callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
        if settings['stage'] == "fit":
            trainer = L.Trainer(max_epochs=10, callbacks=callbacks, log_every_n_steps=5)
            trainer.fit(model, data)
        elif settings['stage'] == "test":
            model = DLClassificationModel.load_from_checkpoint(settings['checkpoint'])
            trainer = L.Trainer()
            trainer.test(model, data)