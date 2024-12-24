from pathlib import Path
from dummy_problems.dataloaders import LettersDataModule
import lightning as L
import torch
from timm import create_model
import torchmetrics

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
        return torch.optim.AdamW(self.model.parameters(), lr=0.005)   


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class SupportVectorMachine:
    def __init__(self, settings):
        self.settings = settings

        # Create pipeline with scaling and SVM and setup a grid search
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf'))
        ])

        param_grid = {
            'svm__C': [0.1, 1, 10],  # Regularization parameter
            'svm__gamma': ['scale', 'auto', 0.1, 0.01]  # Kernel coefficient
        }

        self.grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )

    def __preprocess_data(self, data):
        # TODO: For image data, need to reshape to 2D array
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        return X_train_flat, X_test_flat

    def fit(self, data):
        X, y = self.__preprocess_data(data)

        self.grid_search.fit(X, y)
        
        print(f"Best parameters: {self.grid_search.best_params_}")
        print(f"Best cross-validation score: {self.grid_search.best_score_:.3f}")

    def test(self):
        X, y = self.__preprocess_data(data)

        self.grid_search.score(X, y)

        test_score = self.grid_search.score(X_test_flat, y_test)
        print(f"Test accuracy: {test_score:.3f}")

MODEL_TYPES = {
    "DL": DLClassificationModel,
    "SVM": SupportVectorMachine,
}

# NOTE: DL models only.
# DL models from the "The timm (PyTorch Image Models) Leaderboard" (https://huggingface.co/spaces/timm/leaderboard) 
# have been chosen based on their avg_top1 scores and by searching for *tiny* models only, since the training dataset is small.
MODEL_NAMES = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k",
}

if __name__ == "__main__":
    settings = {
        "dataset_dir": Path("/home/ubuntu/data/letters_dataset"),
        "num_workers": 15,
        "model_type": "DL",
        "model_name": "tiny_vit_21m_512.dist_in22k_ft_in1k",
        "stage": "train",
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=99-step=700.ckpt"
        # "checkpoint": "lightning_logs/version_2/checkpoints/epoch=999-step=7000.ckpt"
    }
    
    data = LettersDataModule(settings)
    model = MODEL_TYPES[settings['model_type']](settings)

    if settings['model_type'] == "DL":
        if settings['stage'] == "train":
            callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", mode="min")]
            trainer = L.Trainer(max_epochs=1000, callbacks=callbacks, log_every_n_steps=5)
            trainer.fit(model, data)
        elif settings['stage'] == "test":
            trainer = L.Trainer(log_every_n_steps=5)
            trainer.test(model=model, datamodule=data, ckpt_path=settings['checkpoint'])

    elif settings['model_type'] == "SVM":
        if settings['stage'] == "train":
            model.fit(data)
        elif settings['stage'] == "test":
            model.test(data)


