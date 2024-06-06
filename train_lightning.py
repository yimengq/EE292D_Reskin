import torch
import argparse
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import pytorch_lightning as pl

# import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Single Sensor Model
class ReskinModel(pl.LightningModule):
    def __init__(self, lr, ema_beta=0.998, use_ema=True):
        super(ReskinModel, self).__init__()
        self.lr = lr
        self.metric_loss = False
        # self.ema_beta = ema_beta
        # self.use_ema = use_ema

        # self.model = nn.Sequential(
        #     nn.Linear(15, 200),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.15),
        #     nn.Linear(200, 200),
        #     nn.Linear(200, 40),
        #     # nn.Dropout(p=0.15),
        #     nn.Linear(40, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 3)
        # )

        self.model = nn.Sequential(
            nn.Linear(15, 200),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(200, 200),
            nn.Linear(200, 40),
            nn.Linear(40, 200),
            nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(200, 200),
            nn.SiLU(),
            nn.Linear(200, 3)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss_dist, loss_force, loss_total = self.one_step(batch, batch_idx, train=True)
        self.log('train_loss', loss_total)
        self.log('train_loss_dist', loss_dist)
        self.log('train_loss_force', loss_force)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss_total

    def validation_step(self, batch, batch_idx):
        loss_dist, loss_force, loss_total = self.one_step(batch, batch_idx, train=False)
        self.log('val_loss', loss_total)
        self.log('val_loss_dist', loss_dist)
        self.log('val_loss_force', loss_force)
        return loss_total

    def l1loss(self, outputs, targets):
        loss_dist = nn.L1Loss()(outputs[:,:2], targets[:,:2])
        loss_force = nn.L1Loss()(outputs[:,2], targets[:,2])
        return loss_dist, loss_force, loss_dist+loss_force
    
    def loss(self, outputs, targets):
        loss_dist = nn.MSELoss()(outputs[:,:2], targets[:,:2])
        loss_force = nn.MSELoss()(outputs[:,2], targets[:,2])
        return loss_dist, loss_force, loss_dist+loss_force
    
    def one_step(self, batch, batch_idx, train):
        inputs, targets = batch
        outputs = self(inputs)
        loss_dist, loss_force, loss_total = self.loss(outputs, targets)
        # if self.metric_loss:
        #     # Evaluation only, use MSELoss
        #     evalLoss = nn.MSELoss()(outputs, targets)
        #     print("Evaluation result:",evalLoss,'\n\n\n')
        return loss_dist, loss_force, loss_total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss'
            }
        }


"""
csv file format:
timestamp, x_cnc, y_cnc, z_cnc, timestamp_mag,
timestamp_ft, Bx1, By1, Bz1, Bx2, 
By2, Bz2, Bx3, By3, Bz3, 
Bx4, By4, Bz4, Bx5, By5,
Bz5, tx, ty, tz, fx,
fy, fz
"""
class IndentDataset(Dataset):
    def __init__(self, file_path, skip=0):
        # Load the data from the CSV file using numpy
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        # self.norm_vecB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 
        #                           0.1, 0.1, 0.1, 0.1, 0.1, 
        #                           0.1, 0.1, 0.1, 0.1, 0.1])
        self.norm_vecB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 
                                  0.1, 0.1, 0.1, 0.1, 0.1, 
                                  0.1, 0.1, 0.1, 0.1, 0.1])*10
        # self.norm_vecF = np.array([0.1, 0.1, -0.25])
        self.norm_vecF = np.array([1, 1, -1])

        # Extract the input features (B) and the labels (xyF)
        self.B = data[:, 5:20]*self.norm_vecB
        self.xyF = data[:, [1,2,-1]]*self.norm_vecF

        self.skip = skip

        if self.skip > 0:
            self.B = self.B[::self.skip]
            self.xyF = self.xyF[::self.skip]

    def __len__(self):
        return len(self.B)

    def __getitem__(self, idx):
        B_sample = torch.tensor(self.B[idx], dtype=torch.float32)
        xyF_sample = torch.tensor(self.xyF[idx], dtype=torch.float32)
        return B_sample, xyF_sample


def main(fpath, lr=3e-4, n_epochs=1000, batch_size=32, AMP=False, seed=None):
    # -----Torch Settings-----
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium') # Not much speed up
    np.set_printoptions(suppress=True, precision=3)

    # -----Load Dataset-----
    datasets_list = []
    for raw_data_path in fpath:
        datasets_list.append(IndentDataset(raw_data_path, skip=0))
        print("Loaded dataset length: ",len(datasets_list[-1]))

    train_dataset = ConcatDataset(datasets_list[:-1])
    # train_dataset, test_dataset = random_split(full_dataset, [int(len(full_dataset)*0.9), len(full_dataset)-int(len(full_dataset)*0.9)])
    test_dataset = datasets_list[-1]
    # test_dataset.skip = 5  # So that validation is much faster
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_data_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # -----Load Model-----
    model = ReskinModel(lr=lr)
    # compiled_model = torch.compile(model, mode="reduce-overhead") # New in Pytorch 2.0, not quite working yet

    torch.cuda.empty_cache()

    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="Logs",name='',flush_secs=1)
    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-7, patience=n_epochs)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=1,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)

    # train model [mac use accelerator mps, else gpu]
    trainer = pl.Trainer(accelerator='cpu', devices=1, precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=1.0, num_sanity_val_steps=0)
    
    trainer.validate(model=model, dataloaders=test_data_loader)
    trainer.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    # model.metric_loss = True
    trainer.validate(model=model, dataloaders=test_data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input flags for training")
    # parser.add_argument('--n_steps', type=int, default=1000, help='denoising steps')

    args = parser.parse_args()
    # data_fpath_lst = ['./combined_data_20240603_171745.csv', './combined_data_20240603_180522.csv', './combined_data_small.csv']
    data_fpath_lst = [ 
                      './combined_data_20240605_1414.csv',
                      './combined_data_20240605_1436.csv', 
                      './combined_data_20240605_1458.csv', 
                      './combined_data_20240605_1531.csv',
                      './combined_data_20240605_1604.csv',
                        './combined_data_20240603_1603.csv', 
                      './combined_data_20240603_1709.csv', 
                      './combined_data_20240603_1625.csv', 
                      './combined_data_20240521_1948.csv',
                      './combined_data_20240603_1648.csv',
                      ]
    main(data_fpath_lst)