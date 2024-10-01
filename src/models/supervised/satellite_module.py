import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics
from torchmetrics import JaccardIndex,Accuracy,AUROC,F1Score

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.unet3 import UNet_3Plus

import torch.nn.functional as F 

class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                        learning_rate=1e-3, model_params: dict = {}):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        super(ESDSegmentation, self).__init__()
        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate

        # initialize the model based on model_type
        if model_type == 'SegmentationCNN':
            self.model = SegmentationCNN(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type == 'UNet':
            self.model = UNet(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type == 'FCNResnetTransfer':
            self.model = FCNResnetTransfer(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type == 'UNet_3Plus':
            self.model = UNet_3Plus(in_channels=in_channels, n_classes=4)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # initialize the accuracy metrics for the semantic segmentation task
        self.train_jaccard = JaccardIndex(num_classes=4,task='multiclass')
        self.train_accuracy = Accuracy(num_classes=4,task='multiclass')
        self.train_f1score = F1Score(num_classes=4,task='multiclass')
        self.val_accuracy = Accuracy(num_classes=4,task='multiclass')
        self.val_auroc = AUROC(num_classes=4,task='multiclass')
        self.val_f1score = F1Score(num_classes=4,task='multiclass')

        
    def forward(self, X):
        # evaluate self.model
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # mask = mask.long()
        mask = mask.float()  # Convert mask to float
        mask = F.interpolate(mask.unsqueeze(1), size=(400, 400), mode="nearest")
        mask = mask.squeeze(1).long()  # Convert back to long if necessary


        # evaluate batch
        out = self(sat_img)
        
        # calculate cross entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out,mask)
        
        predictions = torch.argmax(out,dim=1)

        # update metrics
        self.train_jaccard.update(predictions,mask)
        self.train_accuracy.update(predictions,mask)
        self.train_f1score.update(predictions,mask)
        
        # log metrics
        metrics = {
            "Train Jaccard Index": self.train_jaccard.compute().item(),
            "Train Accuracy": self.train_accuracy.compute().item(),
            "Train F1 Score": self.train_f1score.compute().item()
        }
        
        self.logger.log_metrics(metrics,self.global_step)
        
        # return loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # mask = mask.long()

        mask = mask.float()  # Convert mask to float
        mask = F.interpolate(mask.unsqueeze(1), size=(400, 400), mode="nearest")
        mask = mask.squeeze(1).long()  # Convert back to long if necessary

        # evaluate batch for validation
        out = self(sat_img)
        
        # calculate cross entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out,mask)
        
        # get the class with the highest probability
        predictions = torch.argmax(out,dim=1)
        prob = torch.softmax(out,dim=1)

        # update metrics
        self.val_accuracy.update(predictions,mask)
        self.val_auroc.update(prob,mask)
        self.val_f1score.update(predictions,mask)
        
        # log metrics
        metrics = {
            "Val Accuracy": self.val_accuracy.compute().item(),
            "Val AUROC": self.val_auroc.compute().item(),
            "Val F1 Score": self.val_f1score.compute().item()
        }
        
        self.logger.log_metrics(metrics,self.global_step)
        
        return loss
    
    def configure_optimizers(self):
        # initialize optimizer with AdamW
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # return optimizer
        return optimizer