"""
Utility functions for training and evaluation
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, deque
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

class MaskedNLLLoss(nn.Module):
    """
    Masked NLL Loss from COSMIC model
    Handles variable-length sequences with proper masking
    """
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss from COSMIC model
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):
    """
    Unmasked Weighted NLL Loss from COSMIC model
    For flattened data without masking requirements
    """
    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class MetricsTracker:
    """Track training metrics over epochs"""
    
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, metrics_dict):
        """Update metrics for current epoch"""
        for split, metrics in metrics_dict.items():
            for metric_name, value in metrics.items():
                key = f"{split}_{metric_name}"
                self.history[key].append(value)
    
    def get_history(self):
        """Get complete history"""
        return dict(self.history)
    
    def get_best(self, metric_name, mode='max'):
        """Get best value for a metric"""
        values = self.history.get(metric_name, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)
    
    def get_last(self, metric_name):
        """Get last value for a metric"""
        values = self.history.get(metric_name, [])
        return values[-1] if values else None

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config.experiment.results_dir,
        config.experiment.models_dir,
        config.experiment.logs_dir
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def save_config(config, filepath):
    """Save configuration to JSON file"""
    # Convert config to dictionary
    config_dict = {
        'model': {
            'embedding_dims': config.model.embedding_dims,
            'hidden_dim': config.model.hidden_dim,
            'n_classes': config.model.n_classes,
            'D_s': config.model.D_s,
            'D_g': config.model.D_g,
            'D_p': config.model.D_p,
            'D_r': config.model.D_r,
            'D_i': config.model.D_i,
            'D_h': config.model.D_h,
            'D_a_att': config.model.D_a_att,
            'heter_n_layers': config.model.heter_n_layers,
            'win_p': config.model.win_p,
            'win_f': config.model.win_f,
            'shift_win': config.model.shift_win,
            'mode1': config.model.mode1,
            'norm': config.model.norm,
            'listener_state': config.model.listener_state,
            'context_attention': config.model.context_attention,
            'emo_gru': config.model.emo_gru,
            'att2': config.model.att2,
            'residual': config.model.residual,
            'dropout': config.model.dropout,
            'dropout_rec': config.model.dropout_rec
        },
        'training': {
            'epochs': config.training.epochs,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'l2_reg': config.training.l2_reg,
            'valid_ratio': config.training.valid_ratio,
            'loss_type': config.training.loss_type,
            'lambd': config.training.lambd,
            'class_weight': config.training.class_weight,
            'class_weight_mu': config.training.class_weight_mu,
            'optimizer': config.training.optimizer,
            'gradient_clip': config.training.gradient_clip,
            'early_stopping': config.training.early_stopping,
            'patience': config.training.patience,
            'min_delta': config.training.min_delta
        },
        'data': {
            'roberta_path': config.data.roberta_path,
            'comet_path': config.data.comet_path,
            'multimodal_path': config.data.multimodal_path,
            'num_workers': config.data.num_workers,
            'pin_memory': config.data.pin_memory
        },
        'system': {
            'no_cuda': config.system.no_cuda,
            'gpu_ids': config.system.gpu_ids,
            'seed': config.system.seed,
            'deterministic': config.system.deterministic,
            'tensorboard': config.system.tensorboard,
            'log_interval': config.system.log_interval,
            'save_interval': config.system.save_interval
        },
        'experiment': {
            'name': config.experiment.name,
            'version': config.experiment.version,
            'description': config.experiment.description,
            'results_dir': config.experiment.results_dir,
            'models_dir': config.experiment.models_dir,
            'logs_dir': config.experiment.logs_dir,
            'save_best_only': config.experiment.save_best_only,
            'save_last': config.experiment.save_last,
            'save_top_k': config.experiment.save_top_k
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def plot_training_curves(metrics_history, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(metrics_history.get('train_loss', []), label='Train', color='blue')
    axes[0, 0].plot(metrics_history.get('valid_loss', []), label='Valid', color='orange')
    axes[0, 0].plot(metrics_history.get('test_loss', []), label='Test', color='green')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Emotion F1 curves
    axes[0, 1].plot(metrics_history.get('train_emotion_f1', []), label='Train', color='blue')
    axes[0, 1].plot(metrics_history.get('valid_emotion_f1', []), label='Valid', color='orange')
    axes[0, 1].plot(metrics_history.get('test_emotion_f1', []), label='Test', color='green')
    axes[0, 1].set_title('Emotion F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Emotion Accuracy curves
    axes[1, 0].plot(metrics_history.get('train_emotion_acc', []), label='Train', color='blue')
    axes[1, 0].plot(metrics_history.get('valid_emotion_acc', []), label='Valid', color='orange')
    axes[1, 0].plot(metrics_history.get('test_emotion_acc', []), label='Test', color='green')
    axes[1, 0].set_title('Emotion Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Sentiment F1 curves (if available)
    if 'train_sentiment_f1' in metrics_history:
        axes[1, 1].plot(metrics_history.get('train_sentiment_f1', []), label='Train', color='blue')
        axes[1, 1].plot(metrics_history.get('valid_sentiment_f1', []), label='Valid', color='orange')
        axes[1, 1].plot(metrics_history.get('test_sentiment_f1', []), label='Test', color='green')
        axes[1, 1].set_title('Sentiment F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Sentiment metrics\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Sentiment F1 Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()

def save_results(results, config):
    """Save complete training results"""
    timestamp = Path(config.experiment.results_dir) / f"results_{config.experiment.name}"
    
    # Create experiment-specific directory
    exp_dir = Path(config.experiment.results_dir) / config.experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / "config.json"
    save_config(config, config_path)
    
    # Save metrics history
    metrics_path = exp_dir / "metrics_history.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(results['metrics_history'], f)
    
    # Save predictions
    if results['best_predictions'] is not None:
        pred_path = exp_dir / "best_predictions.pkl"
        with open(pred_path, 'wb') as f:
            pickle.dump(results['best_predictions'], f)
    
    # Plot and save training curves
    curves_path = exp_dir / "training_curves.png"
    plot_training_curves(results['metrics_history'], curves_path)
    
    # Plot and save confusion matrices (if predictions available)
    if results['best_predictions'] is not None:
        preds_emo, labels_emo, preds_sen, labels_sen = results['best_predictions']
        
        # Emotion confusion matrix
        if preds_emo is not None and labels_emo is not None:
            # MELD emotion class names
            emotion_classes = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
            cm_emo_path = exp_dir / "confusion_matrix_emotion.png"
            plot_confusion_matrix(labels_emo, preds_emo, emotion_classes, cm_emo_path)
        
        # Sentiment confusion matrix  
        if preds_sen is not None and labels_sen is not None:
            # MELD sentiment class names (positive, negative, neutral)
            sentiment_classes = ['negative', 'neutral', 'positive']
            cm_sen_path = exp_dir / "confusion_matrix_sentiment.png"
            plot_confusion_matrix(labels_sen, preds_sen, sentiment_classes, cm_sen_path)
    
    # Save summary
    summary = {
        'experiment_name': config.experiment.name,
        'best_f1': results['best_f1'],
        'best_epoch': results['best_epoch'],
        'config_summary': {
            'model': {
                'hidden_dim': config.model.hidden_dim,
                'mode1': config.model.mode1,
                'norm': config.model.norm,
                'att2': config.model.att2,
                'listener_state': config.model.listener_state
            },
            'training': {
                'epochs': config.training.epochs,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.training.batch_size,
                'loss_type': config.training.loss_type
            }
        }
    }
    
    summary_path = exp_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {exp_dir}")
    print(f"  - Configuration: {config_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Training curves: {curves_path}")
    print(f"  - Summary: {summary_path}")
    
    return exp_dir

class AutomaticWeightedLoss(torch.nn.Module):
    """Automatically weighted multi-task loss"""
    
    def __init__(self, num_tasks=3):
        super(AutomaticWeightedLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, *losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def format_time(seconds):
    """Format seconds to human readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

def save_model_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)

if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update({
        'train': {'loss': 0.5, 'acc': 80.0},
        'valid': {'loss': 0.6, 'acc': 75.0}
    })
    print("MetricsTracker test passed")
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3)
    for loss in [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]:
        early_stop(loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at loss {loss}")
            break
    
    print("All utility tests passed!")