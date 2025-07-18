import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets # Import datasets to trigger download

# Import your custom utility functions and model definitions
import utils
import models # This will implicitly load resnet18 from models.py

# Ensure a dedicated directory for saving logs and checkpoints
SAVE_DIR = '/content/drive/MyDrive/gvr_project/runs'
os.makedirs(SAVE_DIR, exist_ok=True)

# Argument Parser
parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training with GVR or SGD on CIFAR-10/100')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='model architecture (only resnet18 supported)')
parser.add_argument('--datasets', default='CIFAR10', type=str,
                    choices=['CIFAR10', 'CIFAR10_noise', 'CIFAR100', 'CIFAR100_noise'],
                    help='dataset (CIFAR10, CIFAR10_noise, CIFAR100, or CIFAR100_noise)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.05, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=200, type=int, metavar='N',
                    help='print frequency (default: 50)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--cutout', action='store_true', default=True,
                    help='apply cutout augmentation')
parser.add_argument('--noise-ratio', default=0.6, type=float,
                    help='noise ratio for noisy CIFAR10/100 (default: 0.6)')
parser.add_argument('--noise-mode', default='sym', type=str,
                    help='noise mode (sym, asym)')
parser.add_argument('--alpha', default=0.001, type=float,
                    help='weight for GVR gradient variance penalty (default: 0.001)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training (default: 1)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--log-file', default='training_log.txt', type=str,
                    help='name of the log file')
parser.add_argument('--optimizer', default='gvr', type=str, choices=['gvr', 'sgd'],
                    help='optimizer to use (gvr, sgd)')
parser.add_argument('--data-root', default='./datasets/', type=str,
                    help='root directory for datasets')


best_acc = 0 # best test accuracy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(train_loader, model, criterion, optimizer, epoch, device, args, logger):
    """
    Performs one training epoch.
    Handles different optimizer step methods (GVR, SAM, SGD).
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss1_meter = AverageMeter() # Specific to GVR, will be 0 for others
    loss2_meter = AverageMeter() # Specific to GVR, will be 0 for others
    gvr_penalty_meter = AverageMeter() # Specific to GVR, will be 0 for others
    top1_train = AverageMeter()

    model.train() # Ensure model is in train mode for the entire epoch

    end = time.time()
    for i, data in enumerate(train_loader):
        # Data loading: always get two augmented views (img1, img2) and target
        # For SGD,  we will only use img1
        # Check if the dataset is noisy or clean to determine data structure
        if args.datasets in ['CIFAR10', 'CIFAR100']:
             # Clean dataset, data is (img1, img2, target) from DoubleAugmentedDataset
             img1, img2, target = data
        elif args.datasets in ['CIFAR10_noise', 'CIFAR100_noise']:
            # Noisy dataset, data is (img1, img2, target) from cifar_dataset (mode='all')
            img1, img2, target = data
        else:
            raise ValueError("Unsupported dataset mode for training. Expected (img1, img2, target).")


        data_time.update(time.time() - end)

        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        if args.optimizer == 'gvr':
            # GVR optimizer step handles everything internally
            output_metrics = optimizer.step(img1, img2, target, zero_grad=True)
            loss_val = output_metrics["total_loss"]
            loss1_meter.update(output_metrics["loss1"], img1.size(0))
            loss2_meter.update(output_metrics["loss2"], img2.size(0))
            gvr_penalty_meter.update(output_metrics["gvr_penalty"], img1.size(0))

        elif args.optimizer == 'sgd':
            # Standard SGD optimization
            optimizer.zero_grad()
            output = model(img1) # SGD uses one view
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            loss1_meter.update(0.0, img1.size(0)) # Not applicable for SGD
            loss2_meter.update(0.0, img1.size(0)) # Not applicable for SGD
            gvr_penalty_meter.update(0.0, img1.size(0)) # Not applicable for SGD

        losses.update(loss_val, img1.size(0))

        # Calculate training accuracy (model remains in train mode, no temporary eval)
        with torch.no_grad():
            # Removed model.eval() and model.train() from here to avoid BatchNorm interference
            output_for_acc = model(img1)
            _, predicted = torch.max(output_for_acc.data, 1)
            total = target.size(0)
            correct = (predicted == target).sum().item()
            acc = correct / total * 100
            top1_train.update(acc, img1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.write(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Total Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                         f'Loss1 {loss1_meter.val:.4f} ({loss1_meter.avg:.4f})\t'
                         f'Loss2 {loss2_meter.val:.4f} ({loss2_meter.avg:.4f})\t'
                         f'GVR Penalty {gvr_penalty_meter.val:.4f} ({gvr_penalty_meter.avg:.4f})\t'
                         f'Train Acc {top1_train.val:.3f} ({top1_train.avg:.3f})\n')
    return losses.avg, top1_train.avg

def test_epoch(val_loader, model, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    top1 = AverageMeter()
    losses = AverageMeter()

    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)
            loss = criterion(output, targets)

            losses.update(loss.item(), inputs.size(0))

            _, predicted = torch.max(output.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            acc = correct / total * 100
            top1.update(acc, inputs.size(0))

    logger.write(f'Test: [Average Loss {losses.avg:.4f}]\t[Accuracy {top1.avg:.3f}]\n')
    return top1.avg, losses.avg

def main():
    global best_acc
    args = parser.parse_args()

    # Set up logging
    sys.stdout = utils.Logger(os.path.join(SAVE_DIR, args.log_file))
    logger = sys.stdout

    # Print arguments for reproducibility
    logger.write(f"--- Arguments ---\n")
    for arg, value in vars(args).items():
        logger.write(f"{arg}: {value}\n")
    logger.write(f"-----------------\n")

    # Set seed for reproducibility
    utils.set_seed(args.seed)

    # Set device
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        print("Using CPU or default GPU.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dataset Download Trigger ---
    # This ensures CIFAR-10/100 dataset is downloaded by torchvision if not present.
    print(f"Ensuring {args.datasets} dataset is downloaded to {args.data_root}...")
    if args.datasets in ['CIFAR10', 'CIFAR10_noise']:
        datasets.CIFAR10(root=args.data_root, train=True, download=True)
        datasets.CIFAR10(root=args.data_root, train=False, download=True)
    elif args.datasets in ['CIFAR100', 'CIFAR100_noise']:
        datasets.CIFAR100(root=args.data_root, train=True, download=True)
        datasets.CIFAR100(root=args.data_root, train=False, download=True)
    print(f"{args.datasets} dataset check/download complete.")
    # --- End Dataset Download Trigger ---


    # Load model (utils.get_model handles num_classes based on args.datasets)
    model = utils.get_model(args)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Load data
    train_loader, val_loader = utils.get_datasets_cutout(args)

    # Initialize optimizer based on argument
    base_optimizer_class = optim.SGD # SGD will be the base for GVR

    if args.optimizer == 'gvr':
        optimizer = utils.GVR(model.parameters(), base_optimizer_class, model, criterion,
                              alpha=args.alpha, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        print(f"Using GVR optimizer with alpha={args.alpha}")
    elif args.optimizer == 'sgd':
        optimizer = base_optimizer_class(model.parameters(), lr=args.lr,
                                         momentum=args.momentum, weight_decay=args.weight_decay)
        print(f"Using standard SGD optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


    # Learning rate scheduler
    # Scheduler always applies to the base optimizer if the custom optimizer wraps one
    if args.optimizer == 'sgd':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs)


    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nStarting training...\n")
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        logger.write(f"\nEpoch {epoch} started...\n")

        # Train for one epoch
        avg_train_loss, current_train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, device, args, logger)
        train_losses.append(avg_train_loss)
        train_accuracies.append(current_train_acc)
        # Print average train loss and accuracy at the end of the epoch
        logger.write(f'Train: [Average Loss {avg_train_loss:.4f}]\t[Accuracy {current_train_acc:.3f}]\n')

        # Evaluate on validation set
        current_val_acc, avg_val_loss = test_epoch(val_loader, model, criterion, device, logger)
        val_accuracies.append(current_val_acc)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step()
        # Accessing LR might be slightly different for SAM/GVR vs plain SGD
        if args.optimizer == 'sgd':
            logger.write(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        else:
            logger.write(f'Current learning rate: {optimizer.base_optimizer.param_groups[0]["lr"]:.6f}\n')



    logger.write(f"\n--- Training Complete --- \n")

    # Plotting
    epochs_range = range(args.start_epoch, args.epochs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(SAVE_DIR, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"\nPlots saved to {plot_path}")

if __name__ == '__main__':
    main()
