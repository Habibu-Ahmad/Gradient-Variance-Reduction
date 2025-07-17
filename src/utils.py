import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
import random
import os
import time
import models # This will now only contain ResNet18 related code
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os.path
import pickle
from PIL import Image
import json # Added for noise file handling

import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

# Placeholder for AUCMeter. If this is a custom class, ensure its definition is available
# in your project or remove its usage if not strictly necessary for the core functionality.
class AUCMeter(object):
    """
    Placeholder for AUCMeter.
    You might need to provide the actual implementation of this class
    if it's a custom utility for evaluating AUC.
    For now, it's a dummy class to prevent import errors.
    """
    def __init__(self):
        pass
    def reset(self):
        pass
    def add(self, *args):
        pass
    def value(self):
        return 0.0, 0.0, 0.0 # Dummy values

def set_seed(seed=1):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    """
    A simple logger that writes messages to both terminal and a file.
    """
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

################################ Datasets and DataLoaders #######################################

class Cutout:
    """
    Cutout data augmentation.
    Randomly masks out a square region of the input image.
    """
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


def unpickle(file):
    """
    Unpickles a file, typically used for CIFAR-10/100 dataset files.
    """
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    """
    Custom CIFAR-10/100 dataset class supporting noisy labels and different modes.
    """
    def __init__(self, dataset='cifar10', r=0.4, noise_mode='sym', root_dir='./datasets/',
                 transform=None, mode='all', noise_file='cifar10.json', pred=[], probability=[], log=''):

        assert dataset in ['cifar10', 'cifar100'], "Dataset must be 'cifar10' or 'cifar100'."

        self.dataset = dataset
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode #mode 'test', 'all', 'labeled', 'unlabeled'

        # Class transition for asymmetric noise (CIFAR-10/100 specific)
        if self.dataset == 'cifar10':
             self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # CIFAR-10 transitions
             data_folder = 'cifar-10-batches-py'
             train_files = [f'data_batch_{n}' for n in range(1, 6)]
             test_file = 'test_batch'
             num_classes = 10
        elif self.dataset == 'cifar100':
             # Define CIFAR-100 asymmetric noise transition if needed, otherwise default to identity
             # Example (simplified, actual CIFAR-100 asymmetric noise is more complex):
             # Moving classes within superclasses, e.g., 'aquatic mammals' -> 'fish'
             # This requires knowing the superclass structure. For simplicity, let's use
             # a basic transition or indicate that asymmetric noise is not standard for CIFAR100
             # in this implementation.
             # For now, we'll use a dummy transition or assume symmetric noise for CIFAR100
             # if asymmetric noise is not explicitly defined.
             self.transition = {} # Placeholder - define actual transitions if needed
             data_folder = 'cifar-100-python'
             train_files = ['train']
             test_file = 'test'
             num_classes = 100


        self.root_dir = os.path.join(root_dir, data_folder)
        self.noise_file = os.path.join(root_dir, noise_file) # Noise file path relative to the main data_root

        if self.mode=='test':
            test_dic = unpickle(os.path.join(self.root_dir, test_file))
            self.test_data = test_dic['data']
            self.test_data = self.test_data.reshape((-1, 3, 32, 32)) # Use -1 for flexible batch size
            self.test_data = self.test_data.transpose((0, 2, 3, 1))   #(N,32,32,3)
            self.test_label = test_dic['labels']
        else:   #'train' mode
            train_data=[]
            train_label=[]
            for f in train_files:
                dpath = os.path.join(self.root_dir, f)
                data_dic = unpickle(dpath)
                train_data.append(data_dic['data'])
                train_label.extend(data_dic['labels']) # Use extend for lists

            train_data = np.concatenate(train_data)
            train_data = train_data.reshape((-1, 3, 32, 32)) # Use -1 for flexible batch size
            train_data = train_data.transpose((0, 2, 3, 1)) #(N,32,32,3)
            total_samples = len(train_label)

            if os.path.exists(self.noise_file):
                print(f"Loading noisy labels from {self.noise_file}")
                noise_label = json.load(open(self.noise_file,"r"))
                # Ensure the loaded noise labels match the dataset size
                if len(noise_label) != total_samples:
                    print(f"Warning: Noise file size {len(noise_label)} does not match dataset size {total_samples}. Regenerating noise.")
                    noise_label = [] # Reset to regenerate
            else:
                 noise_label = [] # Initialize if file doesn't exist

            if not noise_label: # Inject noise if noise_label is empty (file didn't exist or size mismatch)
                print(f"Injecting {noise_mode} noise with ratio {self.r} for {self.dataset}...")
                idx = list(range(total_samples))
                random.shuffle(idx)
                num_noise = int(self.r * total_samples)
                noise_idx = idx[:num_noise]

                for i in range(total_samples):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            # Symmetric noise: pick a random class different from the true class
                            true_label = train_label[i]
                            # Ensure the noisy label is different from the true label
                            possible_noisy_labels = list(range(num_classes))
                            possible_noisy_labels.remove(true_label)
                            noiselabel = random.choice(possible_noisy_labels)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym' and self.dataset == 'cifar10':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym' and self.dataset == 'cifar100':
                             # Implement CIFAR-100 asymmetric noise here if defined in self.transition
                             # For now, if no specific transition is defined, fallback to symmetric or raise error
                             if not self.transition:
                                 print("Warning: Asymmetric noise requested for CIFAR-100 but no transition defined. Falling back to symmetric noise.")
                                 true_label = train_label[i]
                                 possible_noisy_labels = list(range(num_classes))
                                 possible_noisy_labels.remove(true_label)
                                 noiselabel = random.choice(possible_noisy_labels)
                                 noise_label.append(noiselabel)
                             else:
                                 noiselabel = self.transition.get(train_label[i], train_label[i]) # Use get with default
                                 noise_label.append(noiselabel)
                        else:
                            # Fallback for unknown noise modes or if asym for CIFAR100 is not implemented
                            print(f"Warning: Unknown noise mode '{noise_mode}' or asymmetric noise not fully implemented for {self.dataset}. Using true label.")
                            noise_label.append(train_label[i]) # Use true label

                    else:
                        noise_label.append(train_label[i])

                print(f"Saving noisy labels to {self.noise_file} ...")
                json.dump(noise_label,open(self.noise_file,"w"))
                print("Noise injection complete.")


            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label)==np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)
                    auc,_,_ = auc_meter.value()
                    log.write('Numer of labeled samples:%d    AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            # Two augmented views for GVR optimizer
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            # Two augmented views for GVR optimizer
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            # For 'all' mode, which is likely used for initial training,
            # we also provide two augmented views for consistency with GVR.
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return (img, target)

    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)

class cifar_dataloader():
    """
    DataLoader for CIFAR-10/100, handling training and testing sets.
    """
    def __init__(self, dataset='cifar10', r=0.2, noise_mode='sym', batch_size=256, num_workers=4, cutout=False, root_dir='./datasets/', log='', noise_file=''):
        assert dataset in ['cifar10', 'cifar100'], "Dataset must be 'cifar10' or 'cifar100'."

        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutout = cutout
        self.root_dir = root_dir # Keep the main root_dir here
        self.log = log

        # Determine noise file name based on dataset, noise mode, and ratio
        if noise_file:
            self.noise_file = noise_file # Use provided file name if given
        else:
            self.noise_file = f'{dataset}_noise_{noise_mode}_{r}.json'


        # Normalization values for CIFAR-10/100
        if self.dataset == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
        elif self.dataset == 'cifar100':
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 stats

        if self.cutout:
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_loader(self):
        """
        Returns training and validation DataLoaders.
        """
        train_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                      root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                      noise_file=self.noise_file) # Pass noise_file name
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)

        val_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                    root_dir=self.root_dir, transform=self.transform_test, mode="test",
                                    noise_file=self.noise_file) # Pass noise_file name
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, # Test set usually not shuffled
                                num_workers=self.num_workers, pin_memory=True)
        return train_loader, val_loader

def get_datasets_cutout(args):
    """
    Provides data loaders for CIFAR-10/100 with or without noise, supporting Cutout.
    """
    print ('Using Cutout for data augmentation.' if args.cutout else 'Cutout not used.')
    assert args.datasets in ['CIFAR10', 'CIFAR10_noise', 'CIFAR100', 'CIFAR100_noise'], "This function only supports CIFAR10, CIFAR10_noise, CIFAR100, or CIFAR100_noise datasets."

    # Determine normalization based on dataset
    if args.datasets in ['CIFAR10', 'CIFAR10_noise']:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        base_dataset_class = datasets.CIFAR10
        dataset_name = 'cifar10'
    elif args.datasets in ['CIFAR100', 'CIFAR100_noise']:
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        base_dataset_class = datasets.CIFAR100
        dataset_name = 'cifar100'
    else:
        raise ValueError(f"Unsupported dataset type: {args.datasets}")


    if args.datasets in ['CIFAR10', 'CIFAR100']:
        print (f'Loading {args.datasets} dataset (clean).')
        # For clean datasets, we still provide two augmented views for GVR
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
            Cutout() if args.cutout else transforms.Lambda(lambda x: x) # Apply Cutout conditionally
        ])

        train_dataset = base_dataset_class(root=args.data_root, train=True, download=True) # Use data_root
        # Wrap the original dataset to provide two augmented views
        class DoubleAugmentedDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            def __len__(self):
                return len(self.base_dataset)
            def __getitem__(self, idx):
                img, target = self.base_dataset[idx]
                img1 = self.transform(img)
                img2 = self.transform(img)
                return img1, img2, target

        train_loader = torch.utils.data.DataLoader(
            DoubleAugmentedDataset(train_dataset, train_transform),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            base_dataset_class(root=args.data_root, train=False, transform=transforms.Compose([ # Use data_root
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

    elif args.datasets in ['CIFAR10_noise', 'CIFAR100_noise']:
        print(f'Loading {args.datasets} noisy dataset.')
        # Pass args.data_root to cifar_dataloader and the correct dataset name
        cifar_noise_loader = cifar_dataloader(dataset=dataset_name, r=args.noise_ratio,
                                              noise_mode=args.noise_mode, batch_size=args.batch_size,
                                              num_workers=args.workers, cutout=args.cutout,
                                              root_dir=args.data_root)
        train_loader, val_loader = cifar_noise_loader.get_loader()

    return train_loader, val_loader


def get_model(args):
    """
    Initializes and returns the ResNet18 model for CIFAR-10 or CIFAR-100.
    """
    print('Model: {}'.format(args.arch))

    # Determine number of classes based on dataset
    if args.datasets in ['CIFAR10', 'CIFAR10_noise']:
        num_classes = 10
    elif args.datasets in ['CIFAR100', 'CIFAR100_noise']:
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset for model initialization: {args.datasets}")


    # Ensure that only 'resnet18' is requested
    assert args.arch == 'resnet18', "Only ResNet18 model is supported in this configuration."

    # This will call the ResNet constructor from models.py with the correct num_classes
    model = models.ResNet(models.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

# Functions to disable/enable running stats for BatchNorm layers
# These are crucial for GVR to work correctly by freezing BatchNorm during gradient computation
def disable_running_stats(model):
    """
    Disables running mean and variance updates for BatchNorm layers.
    Sets model to eval mode for BatchNorm layers, while keeping other layers in train mode.
    """
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    """
    Re-enables running mean and variance updates for BatchNorm layers.
    Restores original momentum.
    """
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


################################ GVR Optimizer #######################################

class GVR(torch.optim.Optimizer):
    """
    GVR (Gradient Variance Reduction) Optimizer with Speed Optimization:
    - Subsampled GVR penalty computation
    - Last-layer-only gradient comparison
    - Optional second-order offloading via create_graph=False
    """

    def __init__(self, params, base_optimizer, model, criterion, alpha=0.01, gvr_subset_size=16, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(GVR, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.model = model
        self.criterion = criterion
        self.alpha = alpha
        self.gvr_subset_size = gvr_subset_size  # Max batch size used for GVR penalty
        print(f"GVR optimizer initialized with alpha={self.alpha}, subset_size={self.gvr_subset_size}")

        # Target parameter names for GVR penalty (last layers)
        self.gvr_target_layers = ['conv5_x', 'fc'] # Still targetting the last layers

    def _get_params_by_name(self, names):
        """Helper to filter model parameters by layer/module name."""
        params = []
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in names) and param.requires_grad:
                params.append(param)
        return params

    def step(self, img1, img2, target, zero_grad=True):
        if zero_grad:
            self.zero_grad()

        self.model.train()
        disable_running_stats(self.model)

        # Subsample for GVR penalty (reduce compute cost)
        if img1.size(0) > self.gvr_subset_size:
            idx = torch.randperm(img1.size(0))[:self.gvr_subset_size]
            img1_sub = img1[idx]
            img2_sub = img2[idx]
            target_sub = target[idx]
        else:
            img1_sub, img2_sub, target_sub = img1, img2, target

        # Forward on first view
        output1 = self.model(img1_sub)
        loss1 = self.criterion(output1, target_sub)
        grads1 = torch.autograd.grad(loss1,
                                     self._get_params_by_name(self.gvr_target_layers),
                                     create_graph=False)

        # Forward on second view
        output2 = self.model(img2_sub)
        loss2 = self.criterion(output2, target_sub)
        grads2 = torch.autograd.grad(loss2,
                                     self._get_params_by_name(self.gvr_target_layers),
                                     create_graph=False)

        # GVR penalty over selected parameters
        gvr_penalty = sum(((g1 - g2) ** 2).sum() for g1, g2 in zip(grads1, grads2))

        # Full loss over full batch
        output1_full = self.model(img1)
        output2_full = self.model(img2)
        loss1_full = self.criterion(output1_full, target)
        loss2_full = self.criterion(output2_full, target)
        total_loss = (loss1_full + loss2_full)/2 + self.alpha * gvr_penalty

        enable_running_stats(self.model)
        total_loss.backward()
        self.base_optimizer.step()

        return {
            "loss1": loss1_full.item(),
            "loss2": loss2_full.item(),
            "gvr_penalty": gvr_penalty.item(),
            "total_loss": total_loss.item(),
        }

################################ SAM Optimizer #######################################

class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.
    Based on "Sharpness-Aware Minimization for Efficiently Improving Generalization" by Foret et al.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # The closure should do a full forward-backward pass to compute gradients for the second step
        closure = torch.enable_grad()(closure)

        # First step: perturb weights
        self.first_step(zero_grad=True)
        # Re-evaluate loss and compute gradients for the second step
        loss = closure()
        # Second step: apply sharpness-aware update
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                       )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
