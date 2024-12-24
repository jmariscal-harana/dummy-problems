from pathlib import Path
from dummy_problems.dataloaders import LettersDataModule
import lightning as L
import torch
from timm import create_model
import torchmetrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


class DLClassificationModel(L.LightningModule):
    def __init__(self, settings):
        super().__init__()
        self.model = create_model(settings["model_name"], num_classes=26)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=26)

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        
        self.accuracy(outputs, targets)
        self.log('test_acc_step', self.accuracy)

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy)

    def configure_optimizers(self):
        if settings["model_name"] == "tiny_vit_21m_512.dist_in22k_ft_in1k":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.05)  # lower learning rate for small dataset

        return optimizer


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
            n_jobs=15,
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


MODEL_TYPES = {
    "DL": DLClassificationModel,
    "SVM": SupportVectorMachine,
}

# NOTE: DL model settings only.
# DL models from the "The timm (PyTorch Image Models) Leaderboard" (https://huggingface.co/spaces/timm/leaderboard) 
# have been chosen based on their avg_top1 scores and by searching for *tiny* models only, since the training dataset is small.
SETTINGS_DL = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k": {
        "num_workers": 1,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=99-step=700.ckpt",
    }
}

if __name__ == "__main__":
    settings =  {
        "model_type": "SVM",
        "model_name": "tiny_vit_21m_512.dist_in22k_ft_in1k",
        "dataset_dir": Path("/home/ubuntu/data/letters_dataset"),
        "stage": "train",
    }
    if settings["model_type"] == "DL":
        settings.update(SETTINGS_DL[settings["model_name"]])
    
    data = LettersDataModule(settings)
    model = MODEL_TYPES[settings['model_type']](settings)

    if settings['model_type'] == "DL":
        if settings['stage'] == "fit":
            callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
            trainer = L.Trainer(max_epochs=1000, callbacks=callbacks, log_every_n_steps=5)
            trainer.fit(model, data)
        elif settings['stage'] == "test":
            trainer = L.Trainer(log_every_n_steps=5)
            trainer.test(model=model, datamodule=data, ckpt_path=settings['checkpoint'])

    elif settings['model_type'] == "SVM":
        data.setup("train")
        model.fit(data.letters_train)
        data.setup("test")
        model.test(data.letters_test)
