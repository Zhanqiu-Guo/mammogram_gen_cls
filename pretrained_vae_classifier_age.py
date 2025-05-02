import os
import random
import numpy as np
import wandb
import h5py
import lpips

import warnings
warnings.filterwarnings('ignore')

from accelerate import Accelerator
from huggingface_hub import hf_hub_download
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler
from MammogramDataset import MammogramDataset, BalancedSampler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.utils as vutils
import sys
from torch_transforms import compose_transform
sys.path.append("/gpfs/data/geraslab/Ashen/radgpt")
from src.data import dataloading
import torchvision.transforms as transforms

# Import diffusers for loading pre-trained VAE
from diffusers import AutoencoderKL

from transformers import get_cosine_schedule_with_warmup, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau



config = {
    'bs': 4,
    'lr': 5e-5,
    'wd': 0.01,
    'epochs': 1000,
    'num_classes': 3, 
    'img_size': 256, 
    'seed': 1234,
    'warmup_epochs': 3,
    'use_amp': False
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(seed=config['seed'])

class PretrainedVAEClassifier(nn.Module):
    def __init__(self, num_classes=3, img_size=256):
        super(PretrainedVAEClassifier, self).__init__()
        # Load the pretrained VAE
        self.vae = AutoencoderKL.from_pretrained("Likalto4/mammo40k_healthy-lesion", subfolder="vae")
        # Allow fine-tuning of the VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = True
            
        self.scaling_factor = self.vae.config.scaling_factor
        # The latent channels from the pre-trained VAE (e.g., 4)
        self.latent_channels = self.vae.config.latent_channels

        self.lpips_model = lpips.LPIPS(net='vgg')


        self.flattened_latent_size = 4096  # (latent_channels * latent_height * latent_width)
        self.age_emb_dim = 16
        self.age_embed = nn.Linear(1, self.age_emb_dim)

        input_dim = self.flattened_latent_size + self.age_emb_dim  # 4096 + 16 = 4112

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def encode_view(self, x):
        enc = self.vae.encode(x)
        mu = enc.latent_dist.mean 
        logvar = enc.latent_dist.logvar
        z = enc.latent_dist.sample()
        mu = mu * self.scaling_factor
        z = z * self.scaling_factor
        return mu, logvar, z

    def decode_view(self, z):
        z = z / self.scaling_factor
        dec = self.vae.decode(z)
        return dec.sample

    def forward(self, left_mlo, left_cc, right_mlo, right_cc, age=None):
        mu_left_mlo, logvar_left_mlo, z_left_mlo = self.encode_view(left_mlo)
        mu_left_cc,   logvar_left_cc,   z_left_cc   = self.encode_view(left_cc)
        mu_right_mlo, logvar_right_mlo, z_right_mlo = self.encode_view(right_mlo)
        mu_right_cc,  logvar_right_cc,  z_right_cc  = self.encode_view(right_cc)
        
        recon_left_mlo  = self.decode_view(z_left_mlo)
        recon_left_cc   = self.decode_view(z_left_cc)
        recon_right_mlo = self.decode_view(z_right_mlo)
        recon_right_cc  = self.decode_view(z_right_cc)
        
        batch_size = left_mlo.size(0)

        # pooled_left_mlo = self.classifier_proj(mu_left_mlo.flatten(start_dim=1))
        # pooled_left_cc = self.classifier_proj(mu_left_cc, (1, 1)).view(batch_size, -1)
        # pooled_right_mlo = self.classifier_proj(mu_right_mlo, (1, 1)).view(batch_size, -1)
        # pooled_right_cc = self.classifier_proj(mu_right_cc, (1, 1)).view(batch_size, -1)

        flat_left_mlo = mu_left_mlo.view(batch_size, -1)
        flat_left_cc = mu_left_cc.view(batch_size, -1)
        flat_right_mlo = mu_right_mlo.view(batch_size, -1)
        flat_right_cc = mu_right_cc.view(batch_size, -1)
        
        # proj_left_mlo = self.classifier_proj(flat_left_mlo)
        # proj_left_cc  = self.classifier_proj(flat_left_cc)
        # proj_right_mlo = self.classifier_proj(flat_right_mlo)
        # proj_right_cc  = self.classifier_proj(flat_right_cc)

        if age is not None:
            age_embedding = self.age_embed(age)  # (batch_size, age_emb_dim)
        else:
            # In case no age is passed, we initialize a zero tensor.
            age_embedding = torch.zeros(batch_size, self.age_emb_dim, device=flat_left_mlo.device)

        feat_left_mlo = torch.cat([flat_left_mlo, age_embedding], dim=1)
        feat_left_cc  = torch.cat([flat_left_cc,  age_embedding], dim=1)
        feat_right_mlo = torch.cat([flat_right_mlo, age_embedding], dim=1)
        feat_right_cc  = torch.cat([flat_right_cc,  age_embedding], dim=1)

        pred_left_mlo_logits = self.classifier(feat_left_mlo)
        pred_left_cc_logits  = self.classifier(feat_left_cc)
        pred_right_mlo_logits = self.classifier(feat_right_mlo)
        pred_right_cc_logits  = self.classifier(feat_right_cc)

        pred_left_mlo_probs = F.softmax(pred_left_mlo_logits, dim=1)
        pred_left_cc_probs  = F.softmax(pred_left_cc_logits, dim=1)
        pred_right_mlo_probs = F.softmax(pred_right_mlo_logits, dim=1)
        pred_right_cc_probs  = F.softmax(pred_right_cc_logits, dim=1)
        
        return {
            'reconstructions': (recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc),
            'latent_params': {
                'mu': (mu_left_mlo, mu_left_cc, mu_right_mlo, mu_right_cc),
                'logvar': (logvar_left_mlo, logvar_left_cc, logvar_right_mlo, logvar_right_cc)
            },
            'predictions': {
                'logits': (pred_left_mlo_logits, pred_left_cc_logits, pred_right_mlo_logits, pred_right_cc_logits),
                'probs': (pred_left_mlo_probs, pred_left_cc_probs, pred_right_mlo_probs, pred_right_cc_probs)
            }
        }
    
    def compute_loss(
        self, inputs, outputs, targets,
        lpips_model, 
        l1_weight=1.0,
        lpips_weight=1.0,
        kl_weight=1e-6,
        disc_weight=0.5,
        clf_weight=10000
    ):
        left_mlo, left_cc, right_mlo, right_cc = inputs
        recon_left_mlo, recon_left_cc, recon_right_mlo, recon_right_cc = outputs['reconstructions']
        mu_tuple = outputs['latent_params']['mu']
        logvar_tuple = outputs['latent_params']['logvar']
        pred_left_mlo_logits, pred_left_cc_logits, pred_right_mlo_logits, pred_right_cc_logits = outputs['predictions']['logits']
        left_targets, right_targets = targets

        device = left_mlo.device
        batch_size = left_mlo.size(0)

        l1_loss = (F.l1_loss(recon_left_mlo, left_mlo) +
                F.l1_loss(recon_left_cc, left_cc) +
                F.l1_loss(recon_right_mlo, right_mlo) +
                F.l1_loss(recon_right_cc, right_cc)) * l1_weight

        lpips_loss_val = (lpips_model(recon_left_mlo, left_mlo) +
                        lpips_model(recon_left_cc, left_cc) +
                        lpips_model(recon_right_mlo, right_mlo) +
                        lpips_model(recon_right_cc, right_cc))

        lpips_loss = 0 # LPIPS not used
        # lpips_loss = lpips_loss_val.sum() / batch_size * lpips_weight

        recon_loss_total = l1_loss + lpips_loss

        kl_loss = 0
        for mu, logvar in zip(mu_tuple, logvar_tuple):
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=list(range(1, mu.ndim)))
            kl_loss += torch.sum(kl_div)
        kl_loss = (kl_loss / batch_size) * kl_weight

        clf_loss = (F.cross_entropy(pred_left_mlo_logits, left_targets) +
                    F.cross_entropy(pred_left_cc_logits, left_targets) +
                    F.cross_entropy(pred_right_mlo_logits, right_targets) +
                    F.cross_entropy(pred_right_cc_logits, right_targets)) * clf_weight

        total_loss = recon_loss_total + kl_loss + clf_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss_total.item(),
            'l1_loss': l1_loss.item(),
            'lpips_loss': lpips_loss.item(),
            'kl_loss': kl_loss.item(),
            'clf_loss': clf_loss.item(),
        }
        return total_loss, loss_dict

def log_reconstructions(model, val_loader, epoch, device, n_samples=4):
    model.eval()
    with torch.no_grad():
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

        wandb.log({
            "epoch": epoch,
            "left_reconstructions": wandb.Image(grid_left, caption=f"Epoch {epoch}: Left - Top: Original CC, Below: Reconstructed CC, Next: Original MLO, Bottom: Reconstructed MLO"),
            "right_reconstructions": wandb.Image(grid_right, caption=f"Epoch {epoch}: Right - Top: Original CC, Below: Reconstructed CC, Next: Original MLO, Bottom: Reconstructed MLO")
        })

def compute_auc_for_pairs(true_labels, pred_probs, class_pairs=[(0,1), (0,2), (1,2)]):
    auc_dict = {}
    for a, b in class_pairs:
        indices = np.where((true_labels == a) | (true_labels == b))[0]
        if len(indices) == 0 or len(np.unique(true_labels[indices])) < 2:
            auc_dict[f"auc_{a}_vs_{b}"] = 0
            continue
        binary_labels = (true_labels[indices] == b).astype(int)
        probs = pred_probs[indices, b]
        try:
            auc = roc_auc_score(binary_labels, probs)
        except Exception as e:
            auc = 0
        if auc is None:
            auc = 0
        auc_dict[f"auc_{a}_vs_{b}"] = auc * 100
    return auc_dict

def train(model, train_dataloader, optimizer, scheduler, device, use_amp=False):
    model.train()    
    total_loss_sum = 0.0
    total_recon_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_clf_loss_sum = 0.0
    total_l1_loss_sum = 0.0
    total_lpips_loss_sum = 0.0
    
    total_left_correct = 0
    total_right_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    left_labels_list = []
    left_mlo_probs_list = []
    left_cc_probs_list = []
    right_labels_list = []
    right_mlo_probs_list = []
    right_cc_probs_list = []

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
                lpips_model=model.lpips_model
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
        total_l1_loss_sum += loss_dict['l1_loss']
        total_lpips_loss_sum += loss_dict['lpips_loss']

        batch_size = left_mlo.size(0)
        total_samples += batch_size

        pred_left_mlo_logits, pred_left_cc_logits, pred_right_mlo_logits, pred_right_cc_logits = outputs['predictions']['logits']
        pred_left_mlo_probs, pred_left_cc_probs, pred_right_mlo_probs, pred_right_cc_probs = outputs['predictions']['probs']

        left_pred_mlo_labels = pred_left_mlo_logits.argmax(dim=1)
        left_pred_cc_labels = pred_left_cc_logits.argmax(dim=1)
        right_pred_mlo_labels = pred_right_mlo_logits.argmax(dim=1)
        right_pred_cc_labels = pred_right_cc_logits.argmax(dim=1)
        
        total_left_correct += (left_pred_mlo_labels == left_target).sum().item()
        total_left_correct += (left_pred_cc_labels == left_target).sum().item()
        total_right_correct += (right_pred_mlo_labels == right_target).sum().item()
        total_right_correct += (right_pred_cc_labels == right_target).sum().item()

        left_labels_list.append(left_target.cpu().numpy())
        left_mlo_probs_list.append(pred_left_mlo_probs.cpu().detach().numpy())
        left_cc_probs_list.append(pred_left_cc_probs.cpu().detach().numpy())
        right_labels_list.append(right_target.cpu().numpy())
        right_mlo_probs_list.append(pred_right_mlo_probs.cpu().detach().numpy())
        right_cc_probs_list.append(pred_right_cc_probs.cpu().detach().numpy())

    left_acc = total_left_correct / total_samples / 2 * 100
    right_acc = total_right_correct / total_samples / 2 * 100

    left_mlo_all_probs = np.concatenate(left_mlo_probs_list, axis=0)
    left_cc_all_probs = np.concatenate(left_cc_probs_list, axis=0)
    right_mlo_all_probs = np.concatenate(right_mlo_probs_list, axis=0)
    right_cc_all_probs = np.concatenate(right_cc_probs_list, axis=0)

    left_all_labels = np.concatenate(left_labels_list, axis=0)
    right_all_labels = np.concatenate(right_labels_list, axis=0)

    left_mlo_macro_auc = left_cc_macro_auc = right_mlo_macro_auc = right_cc_macro_auc = 0
    try:
        left_mlo_macro_auc = roc_auc_score(left_all_labels, left_mlo_all_probs, multi_class="ovr", average="macro") * 100
        left_cc_macro_auc = roc_auc_score(left_all_labels, left_cc_all_probs, multi_class="ovr", average="macro") * 100
        right_mlo_macro_auc = roc_auc_score(right_all_labels, right_mlo_all_probs, multi_class="ovr", average="macro") * 100
        right_cc_macro_auc = roc_auc_score(right_all_labels, right_cc_all_probs, multi_class="ovr", average="macro") * 100
    except Exception as e:
        print(f"Error computing AUC: {e}")
    wandb.log({
        "train/total_loss": total_loss_sum / len(train_dataloader),
        "train/recon_loss": total_recon_loss_sum / len(train_dataloader),
        "train/kl_loss": total_kl_loss_sum / len(train_dataloader),
        "train/clf_loss": total_clf_loss_sum / len(train_dataloader),
        "train/l1_loss": total_l1_loss_sum / len(train_dataloader),
        "train/lpips_loss": total_lpips_loss_sum / len(train_dataloader),
        "train/left_acc": left_acc,
        "train/right_acc": right_acc,
        "train/accuracy": (left_acc + right_acc) / 2,
        "train/left_mlo_macro_auc": left_mlo_macro_auc,
        "train/left_cc_macro_auc": left_cc_macro_auc,
        "train/right_mlo_macro_auc": right_mlo_macro_auc,
        "train/right_cc_macro_auc": right_cc_macro_auc,
    })

    print(f"Train Loss: {(total_loss_sum / len(train_dataloader)):.4f}, Train Acc: {((left_acc + right_acc) / 2):.2f}%", flush=True)

def validate(model, val_dataloader, scheduler, device, use_amp=False):
    model.eval()
    total_loss_sum = 0.0
    total_recon_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_clf_loss_sum = 0.0
    total_l1_loss_sum = 0.0
    total_lpips_loss_sum = 0.0
    
    total_left_correct = 0
    total_right_correct = 0
    total_samples = 0

    left_labels_list = []
    left_mlo_probs_list = []
    left_cc_probs_list = []
    right_mlo_probs_list = []
    right_cc_probs_list = []
    right_labels_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
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
                    lpips_model=model.lpips_model
                )
            # scheduler.step(loss)
            
            total_loss_sum += loss.item()
            total_recon_loss_sum += loss_dict['recon_loss']
            total_kl_loss_sum += loss_dict['kl_loss']
            total_clf_loss_sum += loss_dict['clf_loss']
            total_l1_loss_sum += loss_dict['l1_loss']
            total_lpips_loss_sum += loss_dict['lpips_loss']
            
            total_samples += current_batch_size
            
            pred_left_mlo_logits, pred_left_cc_logits, pred_right_mlo_logits, pred_right_cc_logits = outputs['predictions']['logits']
            pred_left_mlo_probs, pred_left_cc_probs, pred_right_mlo_probs, pred_right_cc_probs = outputs['predictions']['probs']

            left_pred_mlo_labels = pred_left_mlo_logits.argmax(dim=1)
            left_pred_cc_labels = pred_left_cc_logits.argmax(dim=1)
            right_pred_mlo_labels = pred_right_mlo_logits.argmax(dim=1)
            right_pred_cc_labels = pred_right_cc_logits.argmax(dim=1)
            
            total_left_correct += (left_pred_mlo_labels == left_target).sum().item()
            total_left_correct += (left_pred_cc_labels == left_target).sum().item()
            total_right_correct += (right_pred_mlo_labels == right_target).sum().item()
            total_right_correct += (right_pred_cc_labels == right_target).sum().item()

            left_labels_list.append(left_target.cpu().numpy())
            left_mlo_probs_list.append(pred_left_mlo_probs.cpu().detach().numpy())
            left_cc_probs_list.append(pred_left_cc_probs.cpu().detach().numpy())
            right_labels_list.append(right_target.cpu().numpy())
            right_mlo_probs_list.append(pred_right_mlo_probs.cpu().detach().numpy())
            right_cc_probs_list.append(pred_right_cc_probs.cpu().detach().numpy())

    left_acc = total_left_correct / total_samples / 2 * 100
    right_acc = total_right_correct / total_samples/ 2 * 100


    left_mlo_all_probs = np.concatenate(left_mlo_probs_list, axis=0)
    left_cc_all_probs = np.concatenate(left_cc_probs_list, axis=0)
    right_mlo_all_probs = np.concatenate(right_mlo_probs_list, axis=0)
    right_cc_all_probs = np.concatenate(right_cc_probs_list, axis=0)

    left_all_labels = np.concatenate(left_labels_list, axis=0)
    right_all_labels = np.concatenate(right_labels_list, axis=0)

    # left_mlo_auc = compute_auc_for_pairs(left_mlo_all_labels, left_all_probs)
    # left_cc_auc = compute_auc_for_pairs(left_cc_all_labels, left_all_probs)
    # right_mlo_auc = compute_auc_for_pairs(right_mlo_all_labels, right_all_probs)
    # right_cc_auc = compute_auc_for_pairs(right_cc_all_labels, right_all_probs)

    left_mlo_macro_auc = left_cc_macro_auc = right_mlo_macro_auc = right_cc_macro_auc = 0
    try:
        left_mlo_macro_auc = roc_auc_score(left_all_labels, left_mlo_all_probs, multi_class="ovr", average="macro") * 100
        left_cc_macro_auc = roc_auc_score(left_all_labels, left_cc_all_probs, multi_class="ovr", average="macro") * 100
        right_mlo_macro_auc = roc_auc_score(right_all_labels, right_mlo_all_probs, multi_class="ovr", average="macro") * 100
        right_cc_macro_auc = roc_auc_score(right_all_labels, right_cc_all_probs, multi_class="ovr", average="macro") * 100
    except Exception as e:
        print(f"Error computing AUC: {e}")

    wandb.log({
        "val/total_loss": total_loss_sum / len(val_dataloader),
        "val/recon_loss": total_recon_loss_sum / len(val_dataloader),
        "val/kl_loss": total_kl_loss_sum / len(val_dataloader),
        "val/clf_loss": total_clf_loss_sum / len(val_dataloader),
        "val/l1_loss": total_l1_loss_sum / len(val_dataloader),
        "val/lpips_loss": total_lpips_loss_sum / len(val_dataloader),
        "val/left_acc": left_acc,
        "val/right_acc": right_acc,
        "val/accuracy": (left_acc + right_acc) / 2,
        # "val/left_auc_ub": left_auc.get("auc_0_vs_1", None),
        # "val/left_auc_um": left_auc.get("auc_0_vs_2", None),
        # "val/left_auc_bm": left_auc.get("auc_1_vs_2", None),
        # "val/right_auc_ub": right_auc.get("auc_0_vs_1", None),
        # "val/right_auc_um": right_auc.get("auc_0_vs_2", None),
        # "val/right_auc_bm": right_auc.get("auc_1_vs_2", None),
        "val/left_mlo_macro_auc": left_mlo_macro_auc,
        "val/left_cc_macro_auc": left_cc_macro_auc,
        "val/right_mlo_macro_auc": right_mlo_macro_auc,
        "val/right_cc_macro_auc": right_cc_macro_auc,
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

    model = PretrainedVAEClassifier(
        num_classes=config['num_classes'],
        img_size=config['img_size']
    )

    train_data_file = '/gpfs/data/geraslab/zg2238/data/train_mammogram.pkl'
    valid_data_file = '/gpfs/data/geraslab/zg2238/data/balanced_val_mammogram.pkl'
    with open(train_data_file, 'rb') as f:
        train_list = pickle.load(f)
    with open(valid_data_file, 'rb') as f:
        valid_list = pickle.load(f)

    train_transform = compose_transform(augmentation="standard", resize=(config['img_size'], config['img_size']), image_format="greyscale")

    val_transform = compose_transform(augmentation=None, resize=(config['img_size'], config['img_size']), image_format="greyscale")

    train_dataset = MammogramDataset(train_list, transform=train_transform)
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

    valid_dataset = MammogramDataset(valid_list, transform=val_transform)
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

    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    device = accelerator.device
    best_val_loss = float('inf')

    for iteration in range(config['epochs']):
        # if iteration <= 3:
        #     for name, param in model.named_parameters():
        #         if 'vae' in name:  
        #             param.requires_grad = False
        # else:
        #     for param in model.parameters():
        #         param.requires_grad = True
        
        train(model, train_loader, optimizer, scheduler, device, use_amp=config['use_amp'])
        print("Train Epoch:", iteration)
        val_loss = validate(model, valid_loader, scheduler, device, use_amp=config['use_amp'])
        print("Validation Epoch:", iteration)

        wandb.log({
            "iteration": iteration,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        if iteration % 3 == 0:
            log_reconstructions(model, valid_loader, iteration, device)
        # log_latent_interpolations(model, valid_loader, iteration, device)

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                '/gpfs/scratch/zg2238/mammogram_test/model_checkpoints/best_pretrained_vae_classifier_age.pth'
            )

    wandb.finish()

if __name__ == "__main__":
    run()