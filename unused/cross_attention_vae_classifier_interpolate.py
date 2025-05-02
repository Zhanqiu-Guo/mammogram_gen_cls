import os
import random
import numpy as np
import wandb
import h5py

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

from accelerate import Accelerator
import pickle
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
# from torchvision.models import resnet50, ResNet50_Weights  # No longer needed

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.utils as vutils

import sys
sys.path.append("/gpfs/data/geraslab/Ashen/radgpt")

from src.data import dataloading, torch_transforms
from MammogramDataset import MammogramDataset, BalancedSampler
import torchvision.transforms as transforms

import math
from numbers import Number
from transformers import get_cosine_schedule_with_warmup, AdamW


config = {
    'bs': 32,  # Adjusted batch size.  Smaller for potentially larger images.
    'lr': 8e-4,  # Adjusted learning rate.
    'wd': 0.01,  # weight decay
    'epochs': 1000,  # Adjusted number of epochs.  Start lower, adjust as needed.
    'beta': 0.1,  # beta parameter for TC-VAE
    'latent_dim': 512,  # Reduced latent dimension.
    'num_classes': 3,  # Output classes: 0, 1, 2
    'img_size': 224,  # Increased image size (adjust as needed, and make sure transforms match)
    'seed': 1234,
    'warmup_epochs': 5, # Add warmup epochs
    'use_amp': True  # Use automatic mixed precision
}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important for reproducibility when input sizes vary.

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class LatentCrossAttention(nn.Module):
    def __init__(self, latent_dim):
        super(LatentCrossAttention, self).__init__()
        self.query_fc = nn.Linear(latent_dim, latent_dim)
        self.key_fc = nn.Linear(latent_dim, latent_dim)
        self.value_fc = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mu1, mu2):
        stacked = torch.stack([mu1, mu2], dim=1)  # shape: (N, 2, latent_dim)
        query = self.query_fc(stacked)  # (N, 2, latent_dim)
        key = self.key_fc(stacked)      # (N, 2, latent_dim)
        value = self.value_fc(stacked)  # (N, 2, latent_dim)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1)) # (N, 2, 2)
        attn_weights = self.softmax(scores)
        fused = torch.matmul(attn_weights, value) # (N, 2, latent_dim)
        fused = fused.mean(dim=1) # (N, latent_dim)
        return fused

class MultiVAEClassifier(nn.Module):
    def __init__(self, latent_dim=512, num_classes=3, img_size=224):
        super(MultiVAEClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        resnet = models.resnet18(pretrained=True)
        self.shared_encoder = nn.Sequential(*list(resnet.children())[:-2])
        # (N, 512, img_size/32, img_size/32)
        self.feature_size = img_size // 32
        self.encoder_output_size = 512 * self.feature_size * self.feature_size

        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)

        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, self.encoder_output_size),
            nn.ReLU()
        )
        self.decoder_backbone = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shared_view_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.age_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.latent_cross_attention = LatentCrossAttention(latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def encode_image(self, x):
        features = self.shared_encoder(x)  # (N, 512, feature_size, feature_size)
        flat = features.view(features.size(0), -1)  # (N, encoder_output_size)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_dense(z)  # (N, encoder_output_size)
        x = x.view(-1, 512, self.feature_size, self.feature_size)
        x = self.decoder_backbone(x)
        x = self.shared_view_decoder(x)
        return x

    def forward(self, left_mlo, left_cc, right_mlo, right_cc, age=None):
        mu_left_mlo, logvar_left_mlo = self.encode_image(left_mlo)
        mu_left_cc, logvar_left_cc     = self.encode_image(left_cc)
        mu_right_mlo, logvar_right_mlo = self.encode_image(right_mlo)
        mu_right_cc, logvar_right_cc   = self.encode_image(right_cc)

        z_left_mlo = self.reparameterize(mu_left_mlo, logvar_left_mlo)
        z_left_cc  = self.reparameterize(mu_left_cc, logvar_left_cc)
        z_right_mlo = self.reparameterize(mu_right_mlo, logvar_right_mlo)
        z_right_cc  = self.reparameterize(mu_right_cc, logvar_right_cc)

        recon_left_mlo = self.decode(z_left_mlo)
        recon_left_cc  = self.decode(z_left_cc)
        recon_right_mlo = self.decode(z_right_mlo)
        recon_right_cc  = self.decode(z_right_cc)

        fused_left_mu = self.latent_cross_attention(mu_left_mlo, mu_left_cc)
        fused_right_mu = self.latent_cross_attention(mu_right_mlo, mu_right_cc)

        if age is not None:
            if age.dim() == 1:
                age = age.unsqueeze(1)
            age_feat = self.age_encoder(age)
            fused_left = torch.cat([fused_left_mu, age_feat], dim=1)
            fused_right = torch.cat([fused_right_mu, age_feat], dim=1)
        else:
            fused_left = fused_left_mu
            fused_right = fused_right_mu

        pred_left = self.classifier(fused_left)
        pred_right = self.classifier(fused_right)

        return {
            'reconstructions': (recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc),
            'latent_params': {
                'mu': (mu_left_mlo, mu_left_cc, mu_right_mlo, mu_right_cc),
                'logvar': (logvar_left_mlo, logvar_left_cc, logvar_right_mlo, logvar_right_cc)
            },
            'predictions': (pred_left, pred_right)
        }

    def compute_loss(self, inputs, outputs, targets, recon_weight=100, kl_weight=0.1, clf_weight=1.0):
        left_mlo, left_cc, right_mlo, right_cc = inputs
        recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc = outputs['reconstructions']
        mu_tuple = outputs['latent_params']['mu']
        logvar_tuple = outputs['latent_params']['logvar']
        pred_left, pred_right = outputs['predictions']

        recon_loss = (F.mse_loss(recon_left_mlo, left_mlo) +
                      F.mse_loss(recon_left_cc, left_cc) +
                      F.mse_loss(recon_right_mlo, right_mlo) +
                      F.mse_loss(recon_right_cc, right_cc)) * recon_weight

        batch_size = mu_tuple[0].size(0)
        kl_loss = 0
        for mu, logvar in zip(mu_tuple, logvar_tuple):
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        kl_loss = kl_loss * kl_weight

        left_targets, right_targets = targets
        clf_loss = (F.cross_entropy(pred_left, left_targets) + F.cross_entropy(pred_right, right_targets)) * clf_weight

        total_loss = recon_loss + kl_loss + clf_loss
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'clf_loss': clf_loss.item()
        }
        return total_loss, loss_dict

def log_reconstructions(model, val_loader, epoch, device, n_samples=4):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        data = next(iter(val_loader))
        left_cc = data['left_cc'][:n_samples].to(device)  
        left_mlo = data['left_mlo'][:n_samples].to(device)
        right_cc = data['right_cc'][:n_samples].to(device)
        right_mlo = data['right_mlo'][:n_samples].to(device)
        
        age = data['age'][:n_samples].to(device)
        if age.dim() == 1:
            age = age.unsqueeze(1) 
        
        outputs = model(left_mlo, left_cc, right_mlo, right_cc, age)
        recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc = outputs['reconstructions']

        
        def make_comparison_grid(orig_cc, recon_cc, orig_mlo, recon_mlo):
            return torch.cat([orig_cc, recon_cc, orig_mlo, recon_mlo], dim=0)

        comparison_left = make_comparison_grid(left_cc, recon_left_cc, left_mlo, recon_left_mlo)
        comparison_right = make_comparison_grid(right_cc, recon_right_cc, right_mlo, recon_right_mlo)

        grid_left = vutils.make_grid(comparison_left, nrow=n_samples, normalize=True)
        grid_right = vutils.make_grid(comparison_right, nrow=n_samples, normalize=True)

        # Log images to wandb
        wandb.log({
            "epoch": epoch,
            "left_reconstructions": wandb.Image(grid_left, caption=f"Epoch {epoch}: Left - Top: Original CC, Below: Reconstructed CC, Next: Original MLO, Bottom: Reconstructed MLO"),
            "right_reconstructions": wandb.Image(grid_right, caption=f"Epoch {epoch}: Right - Top: Original CC, Below: Reconstructed CC, Next: Original MLO, Bottom: Reconstructed MLO")
        })


def log_latent_interpolations(model, val_loader, epoch, device, num_interpolations=5):
    model.eval()
    with torch.no_grad():
        interpolation_images = []
        class_pairs = [(0, 1), (0, 2), (1, 2)]

        for class1_idx, class2_idx in class_pairs:
            class1_data = None
            class2_data = None

            for data in val_loader:
                left_labels = data['left_label']
                right_labels = data['right_label']
                
                # Find examples for each class
                class1_mask_left = (left_labels == class1_idx)
                class2_mask_left = (left_labels == class2_idx)
                class1_mask_right = (right_labels == class1_idx)
                class2_mask_right = (right_labels == class2_idx)

                if class1_data is None and torch.any(class1_mask_left):
                    idx = torch.where(class1_mask_left)[0][0]
                    class1_data = {
                        'left_mlo': data['left_mlo'][idx:idx+1].to(device),
                        'left_cc': data['left_cc'][idx:idx+1].to(device),
                        'right_mlo': data['right_mlo'][idx:idx+1].to(device),
                        'right_cc': data['right_cc'][idx:idx+1].to(device),
                        'age': data['age'][idx:idx+1].to(device),
                        'side': 'left'
                    }
                
                if class2_data is None and torch.any(class2_mask_left):
                    idx = torch.where(class2_mask_left)[0][0]
                    class2_data = {
                        'left_mlo': data['left_mlo'][idx:idx+1].to(device),
                        'left_cc': data['left_cc'][idx:idx+1].to(device),
                        'right_mlo': data['right_mlo'][idx:idx+1].to(device),
                        'right_cc': data['right_cc'][idx:idx+1].to(device),
                        'age': data['age'][idx:idx+1].to(device),
                        'side': 'left'
                    }
                
                if class1_data is None and torch.any(class1_mask_right):
                    idx = torch.where(class1_mask_right)[0][0]
                    class1_data = {
                        'left_mlo': data['left_mlo'][idx:idx+1].to(device),
                        'left_cc': data['left_cc'][idx:idx+1].to(device),
                        'right_mlo': data['right_mlo'][idx:idx+1].to(device),
                        'right_cc': data['right_cc'][idx:idx+1].to(device),
                        'age': data['age'][idx:idx+1].to(device),
                        'side': 'right'
                    }
                
                if class2_data is None and torch.any(class2_mask_right):
                    idx = torch.where(class2_mask_right)[0][0]
                    class2_data = {
                        'left_mlo': data['left_mlo'][idx:idx+1].to(device),
                        'left_cc': data['left_cc'][idx:idx+1].to(device),
                        'right_mlo': data['right_mlo'][idx:idx+1].to(device),
                        'right_cc': data['right_cc'][idx:idx+1].to(device),
                        'age': data['age'][idx:idx+1].to(device),
                        'side': 'right'
                    }
                
                if class1_data is not None and class2_data is not None:
                    break

            if class1_data is None or class2_data is None:
                print(f"Warning: Could not find images for classes {class1_idx} and {class2_idx}. Skipping interpolation.")
                continue

            # Encode images to latent space
            mu1, logvar1 = model.encode(
                class1_data['left_mlo'], 
                class1_data['left_cc'],
                class1_data['right_mlo'], 
                class1_data['right_cc'], 
                class1_data['age']
            )
            z1 = model.reparameterize(mu1, logvar1)
            
            mu2, logvar2 = model.encode(
                class2_data['left_mlo'], 
                class2_data['left_cc'],
                class2_data['right_mlo'], 
                class2_data['right_cc'], 
                class2_data['age']
            )
            z2 = model.reparameterize(mu2, logvar2)

            interpolated_images = []
            alphas = np.linspace(0, 1, num_interpolations)

            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc = model.decode(z_interp)
                
                # Choose which side to visualize based on the class data
                if class1_data['side'] == 'left':
                    mlo_img = recon_left_mlo
                    cc_img = recon_left_cc
                else:
                    mlo_img = recon_right_mlo
                    cc_img = recon_right_cc
                
                # Combine views for visualization
                interpolated_images.append(torch.cat([cc_img, mlo_img], dim=0))

            # Get original images
            if class1_data['side'] == 'left':
                class1_cc = class1_data['left_cc']
                class1_mlo = class1_data['left_mlo']
            else:
                class1_cc = class1_data['right_cc']
                class1_mlo = class1_data['right_mlo']
                
            if class2_data['side'] == 'left':
                class2_cc = class2_data['left_cc']
                class2_mlo = class2_data['left_mlo']
            else:
                class2_cc = class2_data['right_cc']
                class2_mlo = class2_data['right_mlo']

            # Create grid with original images on both ends
            grid_images = [torch.cat([class1_cc, class1_mlo], dim=0)]
            grid_images.extend(interpolated_images)
            grid_images.append(torch.cat([class2_cc, class2_mlo], dim=0))
            
            interpolation_grid = torch.cat(grid_images, dim=0)
            interpolation_images.append(interpolation_grid)

        if interpolation_images:
            final_grid = torch.cat(interpolation_images, dim=0)
            grid = vutils.make_grid(final_grid, nrow=num_interpolations+2, normalize=True)

            wandb.log({
                "epoch": epoch,
                "latent_interpolations": wandb.Image(grid, caption=f"Epoch {epoch}: Latent Interpolations between classes")
            })

def compute_auc_for_pairs(true_labels, pred_probs, class_pairs=[(0,1), (0,2), (1,2)]):
    auc_dict = {}
    for a, b in class_pairs:
        indices = np.where((true_labels == a) | (true_labels == b))[0]
        if len(indices) == 0 or len(np.unique(true_labels[indices])) < 2:
            auc_dict[f"auc_{a}_vs_{b}"] = None
            continue
        binary_labels = (true_labels[indices] == b).astype(int)
        probs = pred_probs[indices, b]
        try:
            auc = roc_auc_score(binary_labels, probs)
        except Exception as e:
            auc = None
        auc_dict[f"auc_{a}_vs_{b}"] = auc * 100
    return auc_dict

def train(model, train_dataloader, optimizer, scheduler, device, use_amp=False):
    model.train()    
    total_loss_sum = 0.0
    total_recon_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_clf_loss_sum = 0.0
    
    total_left_correct = 0
    total_right_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    left_labels_list = []
    left_probs_list = []
    right_labels_list = []
    right_probs_list = []

    for batch_idx, batch in enumerate(train_dataloader):
        left_mlo = batch['left_mlo'].to(device)
        left_cc = batch['left_cc'].to(device)
        right_mlo = batch['right_mlo'].to(device)
        right_cc = batch['right_cc'].to(device)
        age = batch['age'].unsqueeze(1).to(device)
        left_target = batch['left_label'].to(device)
        right_target = batch['right_label'].to(device)
        current_batch_size = left_mlo.size(0)        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(left_mlo, left_cc, right_mlo, right_cc, age)
            
            loss, loss_dict = model.compute_loss(
                inputs=(left_mlo, left_cc, right_mlo, right_cc),
                outputs=outputs,
                targets=(left_target, right_target),
                recon_weight=1.0,
                kl_weight=0.1,
                clf_weight=1.0
            )
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss_sum += loss.item()
        total_recon_loss_sum += loss_dict['recon_loss']
        total_kl_loss_sum += loss_dict['kl_loss']
        total_clf_loss_sum += loss_dict['clf_loss']
        
        batch_size = left_mlo.size(0)
        total_samples += batch_size

        pred_left, pred_right = outputs['predictions']
        left_pred_labels = pred_left.argmax(dim=1)
        right_pred_labels = pred_right.argmax(dim=1)
        total_left_correct += (left_pred_labels == left_target).sum().item()
        total_right_correct += (right_pred_labels == right_target).sum().item()
        left_probs = F.softmax(pred_left, dim=1)
        right_probs = F.softmax(pred_right, dim=1)
        left_labels_list.append(left_target.cpu().numpy())
        left_probs_list.append(left_probs.cpu().detach().numpy())
        right_labels_list.append(right_target.cpu().numpy())
        right_probs_list.append(right_probs.cpu().detach().numpy())

    left_acc = total_left_correct / total_samples * 100
    right_acc = total_right_correct / total_samples * 100

    left_all_labels = np.concatenate(left_labels_list, axis=0)
    left_all_probs = np.concatenate(left_probs_list, axis=0)
    right_all_labels = np.concatenate(right_labels_list, axis=0)
    right_all_probs = np.concatenate(right_probs_list, axis=0)

    left_auc = compute_auc_for_pairs(left_all_labels, left_all_probs)
    right_auc = compute_auc_for_pairs(right_all_labels, right_all_probs)

    wandb.log({
        "train/total_loss": total_loss_sum / len(train_dataloader),
        "train/recon_loss": total_recon_loss_sum / len(train_dataloader),
        "train/kl_loss": total_kl_loss_sum / len(train_dataloader),
        "train/clf_loss": total_clf_loss_sum / len(train_dataloader),
        "train/left_acc": left_acc,
        "train/right_acc": right_acc,
        "train/accuracy": (left_acc + right_acc) / 2,
        "train/left_auc_ub": left_auc.get("auc_0_vs_1", None),
        "train/left_auc_um": left_auc.get("auc_0_vs_2", None),
        "train/left_auc_bm": left_auc.get("auc_1_vs_2", None),
        "train/right_auc_ub": right_auc.get("auc_0_vs_1", None),
        "train/right_auc_um": right_auc.get("auc_0_vs_2", None),
        "train/right_auc_bm": right_auc.get("auc_1_vs_2", None),
    })

    print(f"Train Loss: {(total_loss_sum / len(train_dataloader)):.4f}, Train Acc: {((left_acc + right_acc) / 2):.2f}%", flush=True)


def validate(model, val_dataloader, device, use_amp=False):
    model.eval()  # set model to evaluation mode
    total_loss_sum = 0.0
    total_recon_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_clf_loss_sum = 0.0
    
    total_left_correct = 0
    total_right_correct = 0
    total_samples = 0

    left_labels_list = []
    left_probs_list = []
    right_labels_list = []
    right_probs_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Retrieve and move data to device
            left_mlo = batch['left_mlo'].to(device)
            left_cc = batch['left_cc'].to(device)
            right_mlo = batch['right_mlo'].to(device)
            right_cc = batch['right_cc'].to(device)
            age = batch['age'].unsqueeze(1).to(device)
            left_target = batch['left_label'].to(device)
            right_target = batch['right_label'].to(device)
            
            current_batch_size = left_mlo.size(0)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(left_mlo, left_cc, right_mlo, right_cc, age)
                loss, loss_dict = model.compute_loss(
                    inputs=(left_mlo, left_cc, right_mlo, right_cc),
                    outputs=outputs,
                    targets=(left_target, right_target),
                    recon_weight=1.0,
                    kl_weight=0.1,
                    clf_weight=1.0
                )
            
            # Accumulate losses
            total_loss_sum += loss.item()
            total_recon_loss_sum += loss_dict['recon_loss']
            total_kl_loss_sum += loss_dict['kl_loss']
            total_clf_loss_sum += loss_dict['clf_loss']
            
            total_samples += current_batch_size
            
            # Calculate predictions and accuracy
            pred_left, pred_right = outputs['predictions']
            left_pred_labels = pred_left.argmax(dim=1)
            right_pred_labels = pred_right.argmax(dim=1)
            total_left_correct += (left_pred_labels == left_target).sum().item()
            total_right_correct += (right_pred_labels == right_target).sum().item()
            
            # Get probabilities using softmax
            left_probs = F.softmax(pred_left, dim=1)
            right_probs = F.softmax(pred_right, dim=1)
            left_labels_list.append(left_target.cpu().numpy())
            left_probs_list.append(left_probs.cpu().detach().numpy())
            right_labels_list.append(right_target.cpu().numpy())
            right_probs_list.append(right_probs.cpu().detach().numpy())

    # Compute accuracy for left and right predictions
    left_acc = total_left_correct / total_samples * 100
    right_acc = total_right_correct / total_samples * 100

    # Concatenate all predictions and labels for AUC computation
    left_all_labels = np.concatenate(left_labels_list, axis=0)
    left_all_probs = np.concatenate(left_probs_list, axis=0)
    right_all_labels = np.concatenate(right_labels_list, axis=0)
    right_all_probs = np.concatenate(right_probs_list, axis=0)

    # Compute AUC for each pair of classes (using your previously defined function)
    left_auc = compute_auc_for_pairs(left_all_labels, left_all_probs)
    right_auc = compute_auc_for_pairs(right_all_labels, right_all_probs)

    # Log validation statistics to wandb
    wandb.log({
        "val/total_loss": total_loss_sum / len(val_dataloader),
        "val/recon_loss": total_recon_loss_sum / len(val_dataloader),
        "val/kl_loss": total_kl_loss_sum / len(val_dataloader),
        "val/clf_loss": total_clf_loss_sum / len(val_dataloader),
        "val/left_acc": left_acc,
        "val/right_acc": right_acc,
        "val/accuracy": (left_acc + right_acc) / 2,
        "val/left_auc_ub": left_auc.get("auc_0_vs_1", None),
        "val/left_auc_um": left_auc.get("auc_0_vs_2", None),
        "val/left_auc_bm": left_auc.get("auc_1_vs_2", None),
        "val/right_auc_ub": right_auc.get("auc_0_vs_1", None),
        "val/right_auc_um": right_auc.get("auc_0_vs_2", None),
        "val/right_auc_bm": right_auc.get("auc_1_vs_2", None),
    })
    
    print(f"Val Loss: {(total_loss_sum / len(val_dataloader)):.4f}, Val Acc: {((left_acc + right_acc) / 2):.2f}%", flush=True)
    return total_loss_sum / len(val_dataloader)


def run():
    seed_everything(seed=config['seed'])

    wandb.init(
        project="resnet-vae-classifier-mammograms",
        name="resnet-vae-mammogram-training",
        config=config
    )

    accelerator = Accelerator()

    # Initialize model
    model = MultiVAEClassifier(
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes'],
        img_size=config['img_size']
    )

    train_data_file = '/gpfs/data/geraslab/zg2238/data/train_mammogram.pkl'
    valid_data_file = '/gpfs/data/geraslab/zg2238/data/balanced_val_mammogram.pkl'
    with open(train_data_file, 'rb') as f:
        train_list = pickle.load(f)
    with open(valid_data_file, 'rb') as f:
        valid_list = pickle.load(f)

    transform = torch_transforms.compose_transform(augmentation=None, resize=(config['img_size'], config['img_size']), image_format="greyscale")

    train_dataset = MammogramDataset(train_list, transform=transform)
    train_sampler = BalancedSampler(train_dataset, train_list)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['bs'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )

    valid_dataset = MammogramDataset(valid_list, transform=transform)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['bs'],
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )


    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['wd']
    )

    # Scheduler configuration
    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare with accelerator
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    device = accelerator.device
    best_val_loss = float('inf')

    print("Data Loaded")
    num_freeze_epochs = 1
    # Here we train in steps (each step block is 100 training steps)
    for iteration in range(config['epochs']):
        # print(f"\n--- Iteration {iteration} ---", flush=True)
        # if iteration < num_freeze_epochs:
        #     for param in model.encoder.shared_encoder.parameters():
        #         param.requires_grad = False

        # elif iteration == num_freeze_epochs:
        #     print(f"--> Unfreezing model.encoder.shared_encoder weights for epoch {iteration+1} onwards...")
        #     for param in model.encoder.shared_encoder.parameters():
        #         param.requires_grad = True
        train(model, train_loader, optimizer, scheduler, device, use_amp=config['use_amp'])
        print("Train Epoch:", iteration)
        val_loss = validate(model, valid_loader, device, use_amp=config['use_amp'])
        print("Validation Epoch:", iteration)

        wandb.log({
            "iteration": iteration,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Optionally, log reconstructions or latent interpolations at certain intervals
        log_reconstructions(model, valid_loader, iteration, device)
        # log_latent_interpolations(model, valid_loader, iteration, device)

        # Save best model based on validation accuracy
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                '/gpfs/scratch/zg2238/mammogram_test/model_checkpoints/best_vae_classifier_interpolate.pth'
            )

    wandb.finish()

if __name__ == "__main__":
    run()