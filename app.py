import mlflow
import torch
import warnings
from pathlib import Path
from model import ProgrammerModel
from data import ProgrammerDataModule
from pytorch_lightning.cli import LightningCLI

class ProgrammerCLI(LightningCLI):

    def after_fit(self):
        print('Saving model!')
        best_model = self.trainer.checkpoint_callback.best_model_path
        model = ProgrammerModel.load_from_checkpoint(best_model)
        model_dir = Path(self.trainer.default_root_dir).resolve() / 'model'
        model.save(model_dir, self.datamodule.transforms)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
      torch.set_float32_matmul_precision("medium")
      cli = ProgrammerCLI(ProgrammerModel, ProgrammerDataModule)
      