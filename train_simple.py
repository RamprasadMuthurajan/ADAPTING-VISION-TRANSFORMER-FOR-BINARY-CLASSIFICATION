# coding=utf-8
"""
Vision Transformer Training for Binary Classification (Single GPU)
Enhanced with: Full metrics, ROC curves, CSV logging, Early stopping
File: ViT-pytorch/train_simple.py

Usage:
    python train_simple.py \
        --name ants_bees_experiment \
        --data_root my_dataset \
        --pretrained_dir checkpoint/ViT-B_16.npz \
        --early_stopping \
        --patience 10
"""
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from data_utils_simple import get_loader  
logger = logging.getLogger(__name__)


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


class MetricsTracker:
    """Track and save all training metrics"""
    def __init__(self, save_dir, experiment_name):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics_file = os.path.join(save_dir, f"{experiment_name}_metrics.csv")
        self.plots_dir = os.path.join(save_dir, "plots", experiment_name)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize metrics lists
        self.history = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'learning_rate': []
        }
    
    def update(self, step, epoch, train_loss=None, train_acc=None, 
               val_loss=None, val_acc=None, val_auc=None, lr=None):
        """Update metrics"""
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_auc'].append(val_auc)
        self.history['learning_rate'].append(lr)
    
    def save_csv(self):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.history)
        df.to_csv(self.metrics_file, index=False)
        logger.info(f"💾 Saved metrics to {self.metrics_file}")
    
    def plot_metrics(self):
        """Generate and save all plots"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Loss plot
        self._plot_loss()
        
        # 2. Accuracy plot
        self._plot_accuracy()
        
        # 3. AUC plot
        self._plot_auc()
        
        # 4. Learning rate plot
        self._plot_lr()
        
        # 5. Combined plot
        self._plot_combined()
        
        logger.info(f"📊 Saved all plots to {self.plots_dir}")
    
    def _plot_loss(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        
        # Filter out None values
        train_steps = [s for s, l in zip(self.history['step'], self.history['train_loss']) if l is not None]
        train_loss = [l for l in self.history['train_loss'] if l is not None]
        val_steps = [s for s, l in zip(self.history['step'], self.history['val_loss']) if l is not None]
        val_loss = [l for l in self.history['val_loss'] if l is not None]
        
        plt.plot(train_steps, train_loss, label='Training Loss', linewidth=2, alpha=0.8)
        plt.plot(val_steps, val_loss, label='Validation Loss', linewidth=2, alpha=0.8)
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.experiment_name} - Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy(self):
        """Plot training and validation accuracy"""
        plt.figure(figsize=(10, 6))
        
        # Filter out None values
        train_steps = [s for s, a in zip(self.history['step'], self.history['train_acc']) if a is not None]
        train_acc = [a*100 for a in self.history['train_acc'] if a is not None]
        val_steps = [s for s, a in zip(self.history['step'], self.history['val_acc']) if a is not None]
        val_acc = [a*100 for a in self.history['val_acc'] if a is not None]
        
        plt.plot(train_steps, train_acc, label='Training Accuracy', linewidth=2, alpha=0.8)
        plt.plot(val_steps, val_acc, label='Validation Accuracy', linewidth=2, alpha=0.8)
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{self.experiment_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_auc(self):
        """Plot validation AUC"""
        plt.figure(figsize=(10, 6))
        
        val_steps = [s for s, a in zip(self.history['step'], self.history['val_auc']) if a is not None]
        val_auc = [a for a in self.history['val_auc'] if a is not None]
        
        if len(val_auc) > 0:
            plt.plot(val_steps, val_auc, label='Validation AUC-ROC', linewidth=2, alpha=0.8, color='green')
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('AUC-ROC Score', fontsize=12)
            plt.title(f'{self.experiment_name} - AUC-ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim([0.5, 1.0])
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.plots_dir, 'auc_curve.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_lr(self):
        """Plot learning rate schedule"""
        plt.figure(figsize=(10, 6))
        
        steps = [s for s, lr in zip(self.history['step'], self.history['learning_rate']) if lr is not None]
        lrs = [lr for lr in self.history['learning_rate'] if lr is not None]
        
        plt.plot(steps, lrs, linewidth=2, alpha=0.8, color='red')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'{self.experiment_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined(self):
        """Plot combined metrics in subplots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        ax = axes[0, 0]
        train_steps = [s for s, l in zip(self.history['step'], self.history['train_loss']) if l is not None]
        train_loss = [l for l in self.history['train_loss'] if l is not None]
        val_steps = [s for s, l in zip(self.history['step'], self.history['val_loss']) if l is not None]
        val_loss = [l for l in self.history['val_loss'] if l is not None]
        ax.plot(train_steps, train_loss, label='Train', linewidth=2)
        ax.plot(val_steps, val_loss, label='Val', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        train_steps = [s for s, a in zip(self.history['step'], self.history['train_acc']) if a is not None]
        train_acc = [a*100 for a in self.history['train_acc'] if a is not None]
        val_steps = [s for s, a in zip(self.history['step'], self.history['val_acc']) if a is not None]
        val_acc = [a*100 for a in self.history['val_acc'] if a is not None]
        ax.plot(train_steps, train_acc, label='Train', linewidth=2)
        ax.plot(val_steps, val_acc, label='Val', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC
        ax = axes[1, 0]
        val_steps = [s for s, a in zip(self.history['step'], self.history['val_auc']) if a is not None]
        val_auc = [a for a in self.history['val_auc'] if a is not None]
        if len(val_auc) > 0:
            ax.plot(val_steps, val_auc, linewidth=2, color='green')
            ax.set_xlabel('Steps')
            ax.set_ylabel('AUC-ROC')
            ax.set_title('Validation AUC-ROC')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.0])
        
        # Learning Rate
        ax = axes[1, 1]
        steps = [s for s, lr in zip(self.history['step'], self.history['learning_rate']) if lr is not None]
        lrs = [lr for lr in self.history['learning_rate'] if lr is not None]
        ax.plot(steps, lrs, linewidth=2, color='red')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.experiment_name} - Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()


def simple_accuracy(preds, labels):
    """Calculate accuracy"""
    return (preds == labels).mean()


def set_seed(args):
    """Set random seeds for reproducibility"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, model, accuracy, step):
    """Save model checkpoint"""
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_step{step}_acc{accuracy:.4f}.bin")
    torch.save(model.state_dict(), model_checkpoint)
    logger.info(f"💾 Saved model checkpoint to {model_checkpoint}")


def save_metadata(args, best_acc, best_auc, total_steps, training_time):
    """Save training metadata"""
    metadata = {
        'experiment_name': args.name,
        'model_type': args.model_type,
        'dataset': args.data_root,
        'pretrained_weights': args.pretrained_dir,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': training_time,
        'hyperparameters': {
            'img_size': args.img_size,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_steps': args.num_steps,
            'actual_steps': total_steps,
            'warmup_steps': args.warmup_steps,
            'decay_type': args.decay_type,
            'max_grad_norm': args.max_grad_norm,
            'eval_every': args.eval_every,
            'seed': args.seed,
            'early_stopping': args.early_stopping,
            'patience': args.patience if args.early_stopping else None,
        },
        'results': {
            'best_validation_accuracy': float(best_acc),
            'best_validation_auc': float(best_auc) if best_auc > 0 else None,
        },
        'device': str(args.device),
    }
    
    metadata_file = os.path.join(args.output_dir, f"{args.name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"📄 Saved metadata to {metadata_file}")


def plot_roc_curve(y_true, y_scores, save_path, experiment_name):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{experiment_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Saved ROC curve to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, experiment_name, class_names=['Ants', 'Bees']):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'{experiment_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Saved confusion matrix to {save_path}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def setup(args):
    """Prepare model and load pretrained weights"""
    config = CONFIGS[args.model_type]
    num_classes = 2
    
    logger.info(f"⚙️  Building {args.model_type} model for {num_classes} classes")
    
    model = VisionTransformer(
        config, 
        args.img_size, 
        zero_head=True,
        num_classes=num_classes
    )
    
    logger.info(f"📂 Loading pretrained weights from: {args.pretrained_dir}")
    model.load_from(np.load(args.pretrained_dir))
    logger.info(f"✅ Pretrained weights loaded successfully!")
    
    model.to(args.device)
    
    num_params = count_parameters(model)
    logger.info(f"📊 Model Information:")
    logger.info(f"   Architecture:      {args.model_type}")
    logger.info(f"   Hidden size:       {config.hidden_size}")
    logger.info(f"   Num layers:        {config.transformer['num_layers']}")
    logger.info(f"   Num heads:         {config.transformer['num_heads']}")
    logger.info(f"   Total parameters:  {num_params:.2f}M")
    logger.info(f"   Classification:    Linear({config.hidden_size} → {num_classes})")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.info(f"💻 Running on CPU")
    
    return model


def validate(args, model, val_loader, global_step, return_predictions=False):
    """Validation function with full metrics"""
    eval_losses = AverageMeter()
    
    logger.info("🔍 Running Validation...")
    
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = batch
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Forward pass
            logits = model(images)[0]
            loss = loss_fct(logits, labels)
            
            eval_losses.update(loss.item())
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    accuracy = simple_accuracy(all_preds, all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    
    logger.info(f"📈 Validation Results (Step {global_step}):")
    logger.info(f"   Loss:     {eval_losses.avg:.5f}")
    logger.info(f"   Accuracy: {accuracy*100:.2f}%")
    logger.info(f"   AUC-ROC:  {auc:.4f}")
    
    if return_predictions:
        return accuracy, eval_losses.avg, auc, all_labels, all_preds, all_scores
    
    return accuracy, eval_losses.avg, auc


def train(args, model):
    """Train the model with full metrics tracking"""
    import time
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.output_dir, args.name)
    
    # Load dataset
    logger.info("📁 Loading dataset...")
    train_loader, val_loader = get_loader(args)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Print training information
    logger.info("\n" + "="*60)
    logger.info("🚀 STARTING TRAINING")
    logger.info("="*60)
    logger.info(f"Total training steps:  {args.num_steps}")
    logger.info(f"Train batch size:      {args.train_batch_size}")
    logger.info(f"Eval batch size:       {args.eval_batch_size}")
    logger.info(f"Learning rate:         {args.learning_rate}")
    logger.info(f"Warmup steps:          {args.warmup_steps}")
    logger.info(f"Weight decay:          {args.weight_decay}")
    logger.info(f"LR decay type:         {args.decay_type}")
    logger.info(f"Max grad norm:         {args.max_grad_norm}")
    logger.info(f"Image size:            {args.img_size}×{args.img_size}")
    logger.info(f"Eval every:            {args.eval_every} steps")
    logger.info(f"Early stopping:        {args.early_stopping}")
    if args.early_stopping:
        logger.info(f"Patience:              {args.patience} validation checks")
        logger.info(f"Min delta:             {args.min_delta}")
    logger.info(f"Random seed:           {args.seed}")
    logger.info("="*60 + "\n")
    
    model.zero_grad()
    set_seed(args)
    
    losses = AverageMeter()
    global_step = 0
    best_acc = 0.0
    best_auc = 0.0
    epoch = 0
    
    # Early stopping variables
    patience_counter = 0
    early_stopped = False
    
    # Training loop
    while True:
        model.train()
        epoch += 1
        
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training ({global_step}/{t_total})",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True
        )
        
        # Track training accuracy for this epoch
        train_preds, train_labels = [], []
        
        for batch in epoch_iterator:
            images, labels = batch
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Forward pass
            logits = model(images)[0]
            loss = criterion(logits, labels)
            
            # Track predictions for training accuracy
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            losses.update(loss.item())
            global_step += 1
            
            # Get current learning rate
            current_lr = scheduler.get_lr()[0]
            
            # Update progress bar
            epoch_iterator.set_postfix(
                loss=f"{losses.val:.4f}", 
                lr=f"{current_lr:.6f}"
            )
            
            # Log to TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar("train/loss", losses.val, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
            
            # Validation
            if global_step % args.eval_every == 0:
                # Calculate training accuracy for current epoch
                train_acc = simple_accuracy(np.array(train_preds), np.array(train_labels))
                
                # Run validation
                accuracy, val_loss, auc = validate(args, model, val_loader, global_step)
                
                # Log to TensorBoard
                writer.add_scalar("train/accuracy", train_acc, global_step)
                writer.add_scalar("val/accuracy", accuracy, global_step)
                writer.add_scalar("val/loss", val_loss, global_step)
                writer.add_scalar("val/auc", auc, global_step)
                
                # Update metrics tracker
                metrics_tracker.update(
                    step=global_step,
                    epoch=epoch,
                    train_loss=losses.avg,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=accuracy,
                    val_auc=auc,
                    lr=current_lr
                )
                
                # Save CSV after each validation
                metrics_tracker.save_csv()
                
                # Check for improvement
                improved = False
                if accuracy > best_acc + args.min_delta:
                    improvement = accuracy - best_acc
                    best_acc = accuracy
                    best_auc = auc
                    improved = True
                    patience_counter = 0
                    
                    # Save best model
                    save_model(args, model, accuracy, global_step)
                    logger.info(f"🎉 New best accuracy: {best_acc*100:.2f}% (AUC: {best_auc:.4f})")
                else:
                    patience_counter += 1
                    logger.info(f"⚠️  No improvement for {patience_counter}/{args.patience} checks (Best: {best_acc*100:.2f}%)")
                
                # Early stopping check
                if args.early_stopping and patience_counter >= args.patience:
                    logger.info("="*60)
                    logger.info(f"🛑 EARLY STOPPING at step {global_step}, epoch {epoch}")
                    logger.info(f"   Best validation accuracy: {best_acc*100:.2f}%")
                    logger.info(f"   Best validation AUC:      {best_auc:.4f}")
                    logger.info(f"   No improvement for {patience_counter * args.eval_every} steps")
                    logger.info("="*60)
                    early_stopped = True
                    break
                
                # Reset for next epoch
                train_preds, train_labels = [], []
                model.train()
            
            # Update metrics tracker for training loss (every step)
            if global_step % 10 == 0:
                metrics_tracker.update(
                    step=global_step,
                    epoch=epoch,
                    train_loss=losses.avg,
                    lr=current_lr
                )
            
            # Stop if reached total steps
            if global_step >= t_total:
                break
        
        losses.reset()
        
        if early_stopped or global_step >= t_total:
            break
    
    # Final validation with full metrics
    logger.info("\n🏁 Training completed! Running final validation...")
    final_acc, final_loss, final_auc, y_true, y_pred, y_scores = validate(
        args, model, val_loader, global_step, return_predictions=True
    )
    
    # Save final metrics
    metrics_tracker.save_csv()
    
    # Generate all plots
    logger.info("\n📊 Generating plots...")
    metrics_tracker.plot_metrics()
    
    # Plot ROC curve
    roc_path = os.path.join(metrics_tracker.plots_dir, 'roc_curve.png')
    plot_roc_curve(y_true, y_scores, roc_path, args.name)
    
    # Plot confusion matrix
    cm_path = os.path.join(metrics_tracker.plots_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path, args.name)
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=['Ants', 'Bees'])
    report_path = os.path.join(args.output_dir, f"{args.name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"📄 Saved classification report to {report_path}")
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save metadata
    save_metadata(args, best_acc, best_auc, global_step, training_time)
    
    writer.close()
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Training stopped:          {'Early stopping' if early_stopped else 'Reached max steps'}")
    logger.info(f"Best validation accuracy:  {best_acc*100:.2f}%")
    logger.info(f"Best validation AUC:       {best_auc:.4f}")
    logger.info(f"Final validation accuracy: {final_acc*100:.2f}%")
    logger.info(f"Final validation AUC:      {final_auc:.4f}")
    logger.info(f"Final validation loss:     {final_loss:.5f}")
    logger.info(f"Total training steps:      {global_step}")
    logger.info(f"Total epochs:              {epoch}")
    logger.info(f"Training time:             {training_time/60:.1f} minutes")
    logger.info(f"Output directory:          {args.output_dir}")
    logger.info(f"Metrics CSV:               {metrics_tracker.metrics_file}")
    logger.info(f"Plots directory:           {metrics_tracker.plots_dir}")
    logger.info(f"TensorBoard logs:          logs/{args.name}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Vision Transformer Binary Classification with Full Metrics"
    )
    
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Experiment name")
    parser.add_argument("--data_root", required=True,
                        help="Path to dataset root")
    parser.add_argument("--pretrained_dir", required=True,
                        help="Path to pretrained .npz file")
    
    # Model parameters
    parser.add_argument("--model_type", 
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Vision Transformer variant")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="Directory to save outputs")
    
    # Training parameters
    parser.add_argument("--img_size", default=224, type=int,
                        help="Input image resolution")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Validation batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay")
    parser.add_argument("--num_steps", default=2000, type=int,
                        help="Total training steps")
    parser.add_argument("--warmup_steps", default=200, type=int,
                        help="Warmup steps")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="LR decay schedule")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--eval_every", default=50, type=int,
                        help="Validation frequency")
    
    # Early stopping parameters
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum improvement threshold")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    # Print configuration
    logger.info("\n" + "="*60)
    logger.info("🔧 CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Device:           {device}")
    logger.info(f"Model:            {args.model_type}")
    logger.info(f"Dataset:          {args.data_root}")
    logger.info(f"Pretrained:       {args.pretrained_dir}")
    logger.info(f"Experiment name:  {args.name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image size:       {args.img_size}")
    logger.info(f"Batch size:       {args.train_batch_size}")
    logger.info(f"Learning rate:    {args.learning_rate}")
    logger.info(f"Training steps:   {args.num_steps}")
    logger.info(f"Early stopping:   {args.early_stopping}")
    logger.info(f"Random seed:      {args.seed}")
    logger.info("="*60 + "\n")
    
    # Set seed
    set_seed(args)
    
    # Setup model
    model = setup(args)
    
    # Start training
    train(args, model)


if __name__ == "__main__":
    main()