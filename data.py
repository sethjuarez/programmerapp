import math
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import mltable

class ProgrammerDataset(Dataset):
    def __init__(self, path: str):
        tbl = mltable.load(path)
        self.df = tbl.to_pandas_dataframe()

        locations = self.df['location'].unique()
        styles = self.df['style'].unique()
        self.transforms = {
            'location': { locations[i].lower(): i for i in range(len(locations)) },
            'style': { styles[i].lower(): i for i in range(len(styles)) },
            'age': self.minmax('age'),
            'orgsz': self.minmax('orgsz'),
            'yoe': self.minmax('yoe'),
            'projects': self.minmax('projects'),
            'accept': 'yes'
        }

    def minmax(self, col: str):
        return [float(self.df[col].min()), float(self.df[col].max())]

    def get(self, row, col: str):
        xform = self.transforms[col]
        if isinstance(xform, dict):
            return xform[row[col].lower()]
        elif isinstance(xform, list):
            return float(row[col] - xform[0]) / float(xform[1] - xform[0])
        else:
            return 1 if xform == row[col] else 0
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        v = [self.get(row, x) for x in self.transforms.keys()]
        return torch.tensor(v[0]), torch.tensor(v[1]), torch.tensor(v[2:-1], dtype=torch.float32), torch.tensor(v[-1])
    

class ProgrammerDataModule(pl.LightningDataModule):
    def __init__(self, mltable_dir: str = "path/to/dir", 
                       batch_size: int = 32,
                       train_split: float = .8):
        super().__init__()
        self.mltable_dir = mltable_dir
        self.batch_size = batch_size
        self.train_split = train_split

    def setup(self, stage: Optional[str] = None):
        
        self.raw_data = ProgrammerDataset(self.mltable_dir)
        self.transforms = self.raw_data.transforms

        sz = len(self.raw_data)
        train_sz = math.floor(self.train_split * sz)
        val_sz = sz - train_sz
        self.train_dataset, self.val_dataset = random_split(self.raw_data, [train_sz, val_sz])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)