import torch
import torch.nn as nn

import os
import yaml
from tqdm import tqdm
import importlib

import sys
main_path = '/home/syurtseven/GI-motility-quantification/segmentation'
sys.path.append(main_path)

from src.utils.viz_utils import visualize_predictions, plot_metric, plot_train_val_history
from src.utils.metric_utils import compute_dice_score
from src.utils.model_utils import *


class BaseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(BaseUNet, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = CBR(in_channels, 64)
        self.encoder2 = CBR(64, 128)
        self.encoder3 = CBR(128, 256)
        self.encoder4 = CBR(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = CBR(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = CBR(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)


def train_one_epoch(model, 
                    train_loader, 
                    val_loader, 
                    train_loss_history, 
                    val_loss_history, 
                    dice_coef_history, 
                    optimizer, 
                    criterion, 
                    args, 
                    epoch,
                    save_path):
    
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")):
        images, masks = batch
        images, masks = images.to(args.device), masks.to(args.device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    dice_coefficients = 0.0

    with torch.no_grad():
        all_dice_scores = []
        for batch_idx, batch in enumerate(val_loader):

            images, masks = batch
            images, masks = images.to(args.device), masks.to(args.device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            if batch_idx % 500 == 0:
                visualize_predictions(images, masks, outputs, 
                                    save_path=save_path, 
                                    epoch=epoch, 
                                    batch_idx=batch_idx)
            
            preds = (outputs > 0.5).float()
            dice_score = compute_dice_score(masks, preds)
            all_dice_scores.append(dice_score)

        dice_coefficients = sum(all_dice_scores) / len(all_dice_scores)
        dice_coef_history.append(dice_coefficients)

        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"Dice Coefficient for {epoch}: {dice_coefficients}")
        print(f'Epoch {epoch+1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    

def train_model(model, train_loader, val_loader, optimizer, criterion, args, save_path):

    os.makedirs(os.path.join(save_path, 'model'),exist_ok=True)

    train_loss_history = []
    val_loss_history   = []
    dice_coef_history  = []

    for epoch in range(args.epoch):
        train_one_epoch(model, 
                    train_loader, 
                    val_loader, 
                    train_loss_history, 
                    val_loss_history, 
                    dice_coef_history, 
                    optimizer, 
                    criterion, 
                    args, 
                    epoch,
                    save_path)
        
        if (epoch % 2 == 0) and (epoch > 5):
            model_save_path = os.path.join(save_path, 'model', f'model_{epoch}.pt')
            torch.save(model.state_dict(), model_save_path)


    plot_train_val_history(train_loss_history,
                           val_loss_history,
                           save_path,
                           args)

    plot_metric(dice_coef_history, 
                label="dice coeff",
                plot_dir=save_path,
                args=args,
                metric='dice_coeff')
    

def test_model(model, test_loader, criterion, device, save_path, args):
    model.eval()
    test_loss = 0.0
    dice_scores = []

    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = (outputs > 0.5).float()
            dice_score = compute_dice_score(masks, preds)
            dice_scores.append(dice_score)

            if batch_idx % 2 == 0:
                visualize_predictions(images, masks, outputs, 
                                    save_path=save_path, 
                                    epoch='test', 
                                    batch_idx=batch_idx)

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_dice_score = sum(dice_scores) / len(dice_scores)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Dice Score: {avg_dice_score:.4f}")

    plot_metric(dice_scores, 
                label="dice coeff",
                plot_dir=save_path,
                args=args,
                metric='dice_coeff')

    
def get_configs(config_path):

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_class     = get_model_class(config)
    dataset_class   = get_dataset_class(config)
    optimizer_class, optimizer_init_args = get_optimizer_class(config)
    scheduler_class = get_scheduler_class(config)
    criterion  = get_criterion_class(config)

    return model_class, dataset_class, optimizer_class, optimizer_init_args, scheduler_class, criterion
