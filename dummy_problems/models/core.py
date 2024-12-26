from pathlib import Path
from dummy_problems.dataloaders import LettersDataModule
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


class SupportVectorMachine:
    def __init__(self, settings):
        self.settings = settings

        # Create pipeline with scaling and SVM and set up a grid search
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC())
        ])

        param_grid = {
            'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # regularization parameter
            'svm__kernel': ['linear', 'rbf', 'polynomial', 'sigmoid'],
            'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # kernel coefficient
        }

        self.model = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            n_jobs=settings["num_workers"],
            verbose=2
        )

    def __preprocess_data(self, data):
        images = torch.stack([d[0][0] for d in data])
        images = images.reshape(images.shape[0], -1)
        targets = torch.stack([d[1] for d in data])

        return images, targets

    def fit(self, data):
        images, targets = self.__preprocess_data(data)

        self.model.fit(images, targets)
        print(f"Best parameters: {self.model.best_params_}")

    def test(self, data):
        images, targets = self.__preprocess_data(data)
        labels = list(data.labels_to_targets.keys())
        
        predictions = self.model.predict(images)
        
        print(classification_report(targets, predictions, target_names=labels, digits=3))
        print(confusion_matrix(targets, predictions))


class ConvNet(nn.Module):
    def __init__(self, settings):
        super(ConvNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=3,
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
            self.model = create_model(settings["model_name"], num_classes=settings["num_classes"])

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
        if self.settings["model_name"] == "ConvNet":
            optimizer = torch.optim.Adam(self.model.parameters())
        elif self.settings["model_name"] == "tiny_vit_21m_512.dist_in22k_ft_in1k":
            optimizer = torch.optim.AdamW(self.model.parameters())
        else:
            raise NotImplementedError(f"Missing optimizer for {self.settings['model_name']} model")

        return optimizer


MODEL_TYPES = {
    "SVM": SupportVectorMachine,
    "DL": DLClassificationModel,
}

# NOTE: DL model settings only.
# The top DL model from the "The timm (PyTorch Image Models) Leaderboard" (https://huggingface.co/spaces/timm/leaderboard) 
# has been chosen based on its avg_top1 score and by searching for *tiny* models only, since the training dataset is small
# (and I am training on a free CPU instance!).
SETTINGS_DL = {
    "ConvNet": {
        "num_workers": 15,
        "checkpoint": "lightning_logs/version_32/checkpoints/epoch=8-step=5850.ckpt",
    },
    "tiny_vit_21m_512.dist_in22k_ft_in1k": {
        "num_workers": 15,
        "checkpoint": "lightning_logs/version_9/checkpoints/epoch=4-step=35.ckpt",
    },
}

if __name__ == "__main__":
    settings =  {
        "num_classes": 26,
        "dataset_dir": Path("/home/ubuntu/data/letters_dataset"),

        "model_type": "DL",
        "model_name": "tiny_vit_21m_512.dist_in22k_ft_in1k",
        "stage": "fit",
    }
    if settings["model_type"] == "DL":
        settings.update(SETTINGS_DL[settings["model_name"]])
    
    data = LettersDataModule(settings)
    model = MODEL_TYPES[settings['model_type']](settings)

    if settings['model_type'] == "SVM":
        data.setup("train")
        model.fit(data.letters_train)
        data.setup("test")
        model.test(data.letters_test)
    
    elif settings['model_type'] == "DL":
        callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
        if settings['stage'] == "fit":
            trainer = L.Trainer(max_epochs=10, callbacks=callbacks, log_every_n_steps=5)
            trainer.fit(model, data)
        elif settings['stage'] == "test":
            model = DLClassificationModel.load_from_checkpoint(settings['checkpoint'])
            trainer = L.Trainer()
            trainer.test(model, data)