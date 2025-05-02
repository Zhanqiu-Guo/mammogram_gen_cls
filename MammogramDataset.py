from torch.utils.data import Dataset, DataLoader, Sampler
import sys
sys.path.append("/gpfs/data/geraslab/Ashen/radgpt")
from src.data import dataloading, torch_transforms
import torch
import random


class MammogramDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.mammogram_img_dir = "/gpfs/data/geraslab/jp4989/data/2021.07.16.combined_ffdm_cropped/"
        # Store original indices for easy lookup by the sampler if needed,
        # though the sampler primarily works with indices directly.
        self.indices = list(range(len(data_list)))

    def __getitem__(self, idx):
        data = self.data_list[idx]
        images, imagewise_metadata = dataloading.load_mammogram_exam(self.mammogram_img_dir, data, self.transform)

        for image, metadata in zip(images, imagewise_metadata):
             if metadata['laterality'] == 'left':
                 if metadata['view'] == 'MLO': left_mlo = image
                 else: left_cc = image
             else:
                 if metadata['view'] == 'MLO': right_mlo = image
                 else: right_cc = image

        age = torch.tensor(data['age'], dtype=torch.float32)
        cancer_label = data['cancer_label']
        left_benign_label = cancer_label['left_benign']
        right_benign_label = cancer_label['right_benign']
        benign_label = cancer_label['benign']
        left_malignant_label = cancer_label['left_malignant']
        right_malignant_label = cancer_label['right_malignant']
        malignant_label = cancer_label['malignant']
        unknown_label = cancer_label['unknown']

        left_label = 0
        if left_malignant_label: left_label = 2
        elif left_benign_label: left_label = 1

        right_label = 0
        if right_malignant_label: right_label = 2
        elif right_benign_label: right_label = 1

        return {
            'left_mlo': left_mlo, 'left_cc': left_cc,
            'right_mlo': right_mlo, 'right_cc': right_cc,
            'left_images': torch.cat([left_mlo, left_cc], dim=0),
            'right_images': torch.cat([right_mlo, right_cc], dim=0),
            'age': age,
            'left_benign_label': left_benign_label, 'right_benign_label': right_benign_label,
            'benign_label': benign_label, 'left_malignant_label': left_malignant_label,
            'right_malignant_label': right_malignant_label, 'malignant_label': malignant_label,
            'unknown_label': unknown_label, 'left_label': left_label, 'right_label': right_label,
        }

    def __len__(self):
        return len(self.data_list)


class BalancedSampler(Sampler):
    def __init__(self, data_source, data_list):
        self.data_source = data_source
        self.data_list = data_list
        self.malignant_indices = []
        self.benign_indices = []
        self.unknown_indices = []

        for idx, data in enumerate(self.data_list):
            if data['cancer_label']['malignant'] == 1:
                self.malignant_indices.append(idx)
            elif data['cancer_label']['benign'] == 1:
                self.benign_indices.append(idx)
            elif data['cancer_label']['unknown'] == 1:
                self.unknown_indices.append(idx)

        self.num_malignant = len(self.malignant_indices)
        if self.num_malignant == 0:
             print("Warning: No malignant samples found for sampling.")
        self.target_unknown_count = self.num_malignant * 3
        self.target_benign_count = (self.num_malignant) * 2
        self.num_benign_to_sample = min(self.target_benign_count, len(self.benign_indices))
        self.num_unknown_to_sample = min(self.target_unknown_count, len(self.unknown_indices))
        self.num_samples = self.num_malignant + self.num_benign_to_sample + self.num_unknown_to_sample

        print(f"Sampler Initialized: Malignant={self.num_malignant}, "
              f"Sampling Benign={self.num_benign_to_sample} (target {self.target_benign_count}), "
              f"Sampling Unknown={self.num_unknown_to_sample} (target {self.target_unknown_count}). "
              f"Total samples per epoch: {self.num_samples}")

    def __iter__(self):
        print("BalancedSampler: Generating indices for new epoch...")
        sampled_benign = random.sample(self.benign_indices, self.num_benign_to_sample)
        sampled_unknown = random.sample(self.unknown_indices, self.num_unknown_to_sample)
        epoch_indices = self.malignant_indices + sampled_benign + sampled_unknown
        random.shuffle(epoch_indices)

        print(f"BalancedSampler: Yielding {len(epoch_indices)} indices.")
        return iter(epoch_indices)

    def __len__(self):
        return self.num_samples