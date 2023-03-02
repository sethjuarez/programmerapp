import os
import json
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
import pytorch_lightning as pl
import torch.nn.functional as F

class ProgrammerModel(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        
        self.save_hyperparameters()
        self.lr = lr

        self.embed1 = nn.Embedding(7, 4)
        self.embed2 = nn.Embedding(2, 2)
        self.fc1 = nn.Linear(10, 1)
        
    def forward(self, x):
        e1 = self.embed1(x[0])
        e2 = self.embed2(x[1])
        l1 = self.fc1(torch.cat((e1, e2, x[2]), dim=1))
        return torch.sigmoid(l1)
    
    def __compute(self, batch):
        x = batch[0:-1]
        y = batch[-1].unsqueeze(1).to(torch.float32)
        y_hat = self(x)
        
        # loss
        loss = F.binary_cross_entropy(y_hat, y)

        # accuracy 
        _, preds = torch.max(y_hat, 1)
        _, truth = torch.max(y, 1)
        accuracy = torch.sum((preds == truth).float()).item() / len(x[0])
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, acc = self.__compute(batch)
        self.log('loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.__compute(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
    
    def save(self, model_dir: Path, transforms):

        if not model_dir.exists():
            os.makedirs(str(model_dir))

        batch_size = 1
        fake = [torch.randint(7, (batch_size,)), torch.randint(2, (batch_size,)), torch.rand((batch_size, 4))]

        self.to_onnx(model_dir / 'model.onnx', 
                    fake, 
                    export_params=True,
                    input_names=['location', 'style', 'numerics'],
                    output_names=['prediction'])

        file_size = os.path.getsize(str(model_dir / 'model.onnx'))
        param_size = sum(p.numel() for p in self.parameters())

        with open(model_dir / 'model.json', 'w') as f:
            f.write(json.dumps({ 
              'transforms': transforms,
              'params': param_size,
              'size': file_size,
              'timestamp': datetime.now().isoformat()
            }, indent=4))
