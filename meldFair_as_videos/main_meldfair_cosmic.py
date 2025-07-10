#!/usr/bin/env python3
"""
Main script for MELD-FAIR + COSMIC + GraphSmile emotion recognition
Includes training, testing, comprehensive evaluation, and visualization
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from meldfair_video_cosmic_model import MELDFAIRVideoCommonsenseGraphSmile
from meldfair_video_cosmic_dataloader import get_MELDFAIR_VIDEO_COSMIC_loaders


class EmotionRecognitionTrainer:
    """Complete trainer for emotion recognition with evaluation and visualization"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        
        # Emotion mapping
        self.emotion_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        self.sentiment_names = ['neutral', 'positive', 'negative']
        
        # Create directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
        
        # Training history
        self.train_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'dev_loss': [], 'dev_acc': [], 'dev_f1': []
        }
        
    def create_class_weights(self, mu=1):
        """Create class weights for MELD emotion labels"""
        unique = [0, 1, 2, 3, 4, 5, 6]
        labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
        total = np.sum(list(labels_dict.values()))
        weights = []
        for key in unique:
            score = np.log(mu * total / labels_dict[key])
            weights.append(score)
        return torch.FloatTensor(weights)
    
    def load_data(self):
        """Load MELD-FAIR + COSMIC data"""
        print('📊 Loading MELD-FAIR Video + COSMIC data...')
        try:
            self.train_loader, self.dev_loader, self.test_loader = get_MELDFAIR_VIDEO_COSMIC_loaders(
                self.args.meldfair_path, 
                self.args.cosmic_roberta_path, 
                self.args.cosmic_comet_path,
                batch_size=self.args.batch_size, 
                num_workers=self.args.num_workers, 
                pin_memory=not self.args.no_cuda,
                max_frames=self.args.max_frames
            )
            
            print(f'✅ Data loaded successfully!')
            print(f'   Train batches: {len(self.train_loader)}')
            print(f'   Dev batches: {len(self.dev_loader)}')
            print(f'   Test batches: {len(self.test_loader)}')
            return True
            
        except Exception as e:
            print(f'❌ Error loading data: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_model(self):
        """Initialize the model and optimizer"""
        print('🤖 Initializing model...')
        
        # Model
        self.model = MELDFAIRVideoCommonsenseGraphSmile(
            self.args, n_classes_emo=7, n_classes_sent=3
        )
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'   Total parameters: {total_params:,}')
        print(f'   Trainable parameters: {trainable_params:,}')
        
        # Loss function
        if self.args.class_weight:
            loss_weights = self.create_class_weights(self.args.class_weight_mu)
            loss_weights = loss_weights.to(self.device)
            self.loss_function = nn.CrossEntropyLoss(weight=loss_weights)
            print('   Using weighted CrossEntropyLoss')
        else:
            self.loss_function = nn.CrossEntropyLoss()
            print('   Using standard CrossEntropyLoss')
        
        # Optimizer with different learning rates for video components
        video_params = list(self.model.video_frontend.parameters()) + list(self.model.video_temporal.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if not any(name in n for name in ['video_frontend', 'video_temporal'])]
        
        self.optimizer = optim.AdamW([
            {'params': video_params, 'lr': self.args.lr * self.args.video_lr_factor, 'name': 'video'},
            {'params': other_params, 'lr': self.args.lr, 'name': 'other'}
        ], weight_decay=self.args.l2)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=self.args.lr_factor, 
            patience=self.args.lr_patience, verbose=True, min_lr=1e-7
        )
        
        print(f'   Optimizer: AdamW (LR: {self.args.lr}, Video LR: {self.args.lr * self.args.video_lr_factor})')
        print(f'   Scheduler: ReduceLROnPlateau (factor: {self.args.lr_factor}, patience: {self.args.lr_patience})')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses, epoch_preds, epoch_labels = [], [], []
        
        start_time = time.time()
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move batch to device
            self.move_batch_to_device(batch)
            
            try:
                # Forward pass
                emotion_logits, sentiment_logits, shift_logits, fused_features = self.model(batch)
                
                # Calculate loss
                emotion_labels = batch['emotion_labels']
                loss = self.loss_function(emotion_logits, emotion_labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                self.optimizer.step()
                
                # Record metrics
                pred = torch.argmax(emotion_logits, dim=1)
                epoch_preds.extend(pred.cpu().numpy())
                epoch_labels.extend(emotion_labels.cpu().numpy())
                epoch_losses.append(loss.item())
                
                # Progress logging
                if i % self.args.log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f'   Batch {i:3d}/{len(self.train_loader)} | Loss: {loss.item():.4f} | {elapsed:.1f}s')
                    
            except Exception as e:
                print(f'   ⚠️  Error in batch {i}: {e}')
                continue
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        avg_acc = accuracy_score(epoch_labels, epoch_preds) * 100 if epoch_preds else 0
        avg_f1 = f1_score(epoch_labels, epoch_preds, average='weighted') * 100 if epoch_preds else 0
        
        return avg_loss, avg_acc, avg_f1
    
    def evaluate_epoch(self, dataloader, split_name='dev'):
        """Evaluate for one epoch"""
        self.model.eval()
        epoch_losses, epoch_preds, epoch_labels = [], [], []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Move batch to device
                self.move_batch_to_device(batch)
                
                try:
                    # Forward pass
                    emotion_logits, sentiment_logits, shift_logits, fused_features = self.model(batch)
                    
                    # Calculate loss
                    emotion_labels = batch['emotion_labels']
                    loss = self.loss_function(emotion_logits, emotion_labels)
                    
                    # Record metrics
                    pred = torch.argmax(emotion_logits, dim=1)
                    epoch_preds.extend(pred.cpu().numpy())
                    epoch_labels.extend(emotion_labels.cpu().numpy())
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f'   ⚠️  Error in {split_name} batch {i}: {e}')
                    continue
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        avg_acc = accuracy_score(epoch_labels, epoch_preds) * 100 if epoch_preds else 0
        avg_f1 = f1_score(epoch_labels, epoch_preds, average='weighted') * 100 if epoch_preds else 0
        
        return avg_loss, avg_acc, avg_f1, epoch_labels, epoch_preds
    
    def move_batch_to_device(self, batch):
        """Move batch tensors to device"""
        tensor_keys = ['audio', 'video_sequences', 'emotion_labels', 'sentiment_labels', 
                      'audio_mask', 'video_mask']
        
        for key in tensor_keys:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        
        # Move text and commonsense features
        batch['text_features'] = [tf.to(self.device) for tf in batch['text_features']]
        batch['commonsense_features'] = [cf.to(self.device) for cf in batch['commonsense_features']]
    
    def train(self):
        """Main training loop"""
        print(f'\n🚀 Starting training for {self.args.epochs} epochs...')
        print(f'   Device: {self.device}')
        print(f'   Batch size: {self.args.batch_size}')
        print(f'   Max frames: {self.args.max_frames}')
        
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            print(f'\n📈 Epoch {epoch+1}/{self.args.epochs}')
            
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            dev_loss, dev_acc, dev_f1, dev_labels, dev_preds = self.evaluate_epoch(self.dev_loader, 'dev')
            
            # Update learning rate
            self.scheduler.step(dev_f1)
            
            # Record history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['train_f1'].append(train_f1)
            self.train_history['dev_loss'].append(dev_loss)
            self.train_history['dev_acc'].append(dev_acc)
            self.train_history['dev_f1'].append(dev_f1)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'   📊 Results:')
            print(f'      Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%')
            print(f'      Dev   - Loss: {dev_loss:.4f}, Acc: {dev_acc:.2f}%, F1: {dev_f1:.2f}%')
            print(f'      Time: {epoch_time:.1f}s, LR: {current_lr:.2e}')
            
            # Save best model
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_epoch = epoch
                patience_counter = 0
                
                self.save_model(epoch, dev_f1, is_best=True)
                print(f'      🎉 New best F1: {best_f1:.2f}%')
            else:
                patience_counter += 1
                print(f'      ⏳ Patience: {patience_counter}/{self.args.patience}')
            
            # Early stopping
            if patience_counter >= self.args.patience:
                print(f'\n⏹️  Early stopping triggered at epoch {epoch+1}')
                break
            
            print('-' * 60)
        
        print(f'\n✅ Training completed!')
        print(f'   Best F1: {best_f1:.2f}% at epoch {best_epoch+1}')
        
        return best_f1, best_epoch
    
    def save_model(self, epoch, f1_score, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'f1_score': f1_score,
            'args': self.args,
            'train_history': self.train_history
        }
        
        if is_best:
            model_path = os.path.join(self.args.save_dir, f'{self.args.model_name}_best.pth')
            torch.save(checkpoint, model_path)
    
    def load_best_model(self):
        """Load the best model for testing"""
        model_path = os.path.join(self.args.save_dir, f'{self.args.model_name}_best.pth')
        
        if not os.path.exists(model_path):
            print(f'❌ Best model not found at {model_path}')
            return False
        
        print(f'📂 Loading best model from {model_path}')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        return True
    
    def comprehensive_evaluation(self, dataloader, split_name='test'):
        """Perform comprehensive evaluation with detailed metrics"""
        print(f'\n🔍 Comprehensive {split_name} evaluation...')
        
        self.model.eval()
        all_preds, all_labels = [], []
        all_probs = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                self.move_batch_to_device(batch)
                
                try:
                    emotion_logits, _, _, _ = self.model(batch)
                    emotion_labels = batch['emotion_labels']
                    
                    # Get predictions and probabilities
                    probs = torch.softmax(emotion_logits, dim=1)
                    preds = torch.argmax(emotion_logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(emotion_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                except Exception as e:
                    print(f'   ⚠️  Error in evaluation batch {i}: {e}')
                    continue
        
        if not all_preds:
            print('❌ No valid predictions for evaluation')
            return {}
        
        # Calculate comprehensive metrics
        results = self.calculate_detailed_metrics(all_labels, all_preds, all_probs, split_name)
        
        return results
    
    def calculate_detailed_metrics(self, labels, preds, probs, split_name):
        """Calculate detailed evaluation metrics"""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(labels, preds) * 100
        results['balanced_accuracy'] = balanced_accuracy_score(labels, preds) * 100
        results['weighted_f1'] = f1_score(labels, preds, average='weighted') * 100
        results['macro_f1'] = f1_score(labels, preds, average='macro') * 100
        results['micro_f1'] = f1_score(labels, preds, average='micro') * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        results['per_class'] = {}
        for i, emotion in enumerate(self.emotion_names):
            results['per_class'][emotion] = {
                'precision': precision[i] * 100,
                'recall': recall[i] * 100,
                'f1': f1[i] * 100,
                'support': int(support[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        results['confusion_matrix'] = cm.tolist()
        
        # Print results
        self.print_evaluation_results(results, split_name)
        
        # Generate plots
        self.plot_confusion_matrix(cm, split_name)
        
        return results
    
    def print_evaluation_results(self, results, split_name):
        """Print formatted evaluation results"""
        print(f'\n📊 {split_name.upper()} RESULTS:')
        print('=' * 50)
        
        # Overall metrics
        print(f'🎯 Overall Metrics:')
        print(f'   Accuracy:          {results["accuracy"]:.2f}%')
        print(f'   Balanced Accuracy: {results["balanced_accuracy"]:.2f}%')
        print(f'   Weighted F1:       {results["weighted_f1"]:.2f}%')
        print(f'   Macro F1:          {results["macro_f1"]:.2f}%')
        print(f'   Micro F1:          {results["micro_f1"]:.2f}%')
        
        # Per-class metrics
        print(f'\n📈 Per-Class Metrics:')
        print(f'{"Emotion":<10} {"Precision":<10} {"Recall":<10} {"F1":<10} {"Support":<10}')
        print('-' * 50)
        
        for emotion, metrics in results['per_class'].items():
            print(f'{emotion:<10} {metrics["precision"]:<10.2f} {metrics["recall"]:<10.2f} '
                  f'{metrics["f1"]:<10.2f} {metrics["support"]:<10}')
        
        # Best and worst performing emotions
        f1_scores = [(emotion, metrics['f1']) for emotion, metrics in results['per_class'].items()]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f'\n🏆 Best performing: {f1_scores[0][0]} (F1: {f1_scores[0][1]:.2f}%)')
        print(f'⚠️  Worst performing: {f1_scores[-1][0]} (F1: {f1_scores[-1][1]:.2f}%)')
    
    def plot_training_history(self):
        """Plot training history"""
        if not any(self.train_history['train_loss']):
            print('⚠️  No training history to plot')
            return
        
        print('📊 Plotting training history...')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.train_history['dev_loss'], 'r-', label='Dev', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.train_history['dev_acc'], 'r-', label='Dev', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 score plot
        axes[1, 0].plot(epochs, self.train_history['train_f1'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.train_history['dev_f1'], 'r-', label='Dev', linewidth=2)
        axes[1, 0].set_title('Weighted F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        axes[1, 1].text(0.5, 0.5, 'Model Architecture:\n\n'
                        f'Video Frontend: CNN\n'
                        f'Video Temporal: TCN + Self-Attention\n'
                        f'Text: RoBERTa (4 layers)\n'
                        f'Commonsense: COMET (9 features)\n'
                        f'Fusion: GraphSmile HeterGConv\n'
                        f'Classes: 7 emotions',
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 1].set_title('Model Info')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.args.save_dir, 'plots', 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'   💾 Training history saved to {plot_path}')
    
    def plot_confusion_matrix(self, cm, split_name):
        """Plot confusion matrix"""
        print(f'📊 Plotting confusion matrix for {split_name}...')
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=self.emotion_names,
                   yticklabels=self.emotion_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'Confusion Matrix - {split_name.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Add count annotations
        for i in range(len(self.emotion_names)):
            for j in range(len(self.emotion_names)):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                        ha='center', va='center', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.args.save_dir, 'plots', f'confusion_matrix_{split_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'   💾 Confusion matrix saved to {plot_path}')
    
    def save_results(self, train_results, dev_results, test_results):
        """Save all results to files"""
        print('💾 Saving results...')
        
        # Comprehensive results dictionary
        all_results = {
            'model_info': {
                'model_name': self.args.model_name,
                'total_epochs': len(self.train_history['train_loss']),
                'best_dev_f1': max(self.train_history['dev_f1']) if self.train_history['dev_f1'] else 0,
                'final_train_f1': self.train_history['train_f1'][-1] if self.train_history['train_f1'] else 0,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device)
            },
            'hyperparameters': {
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.lr,
                'epochs': self.args.epochs,
                'max_frames': self.args.max_frames,
                'hidden_dim': self.args.hidden_dim,
                'dropout': self.args.drop,
                'use_commonsense': self.args.use_commonsense,
                'commonsense_fusion': self.args.commonsense_fusion,
                'class_weight': self.args.class_weight
            },
            'results': {
                'train': train_results,
                'dev': dev_results,
                'test': test_results
            },
            'training_history': self.train_history
        }
        
        # Save as JSON
        results_path = os.path.join(self.args.save_dir, f'{self.args.model_name}_comprehensive_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save as text summary
        summary_path = os.path.join(self.args.save_dir, f'{self.args.model_name}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("MELD-FAIR + COSMIC + GraphSmile Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"  Model: {self.args.model_name}\n")
            f.write(f"  Parameters: {all_results['model_info']['parameters']:,}\n")
            f.write(f"  Device: {all_results['model_info']['device']}\n")
            f.write(f"  Max Frames: {self.args.max_frames}\n")
            f.write(f"  Batch Size: {self.args.batch_size}\n")
            f.write(f"  Commonsense: {self.args.use_commonsense}\n\n")
            
            f.write("FINAL TEST RESULTS:\n")
            f.write(f"  Accuracy: {test_results['accuracy']:.2f}%\n")
            f.write(f"  Weighted F1: {test_results['weighted_f1']:.2f}%\n")
            f.write(f"  Macro F1: {test_results['macro_f1']:.2f}%\n")
            f.write(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.2f}%\n\n")
            
            f.write("PER-CLASS F1 SCORES:\n")
            for emotion, metrics in test_results['per_class'].items():
                f.write(f"  {emotion}: {metrics['f1']:.2f}%\n")
        
        print(f'   📄 Comprehensive results: {results_path}')
        print(f'   📄 Summary: {summary_path}')


def get_recommended_settings():
    """Get recommended training settings"""
    return {
        'batch_size': 2,  # Small due to video memory requirements
        'epochs': 25,     # Reasonable for initial experiments
        'lr': 0.0001,     # Conservative learning rate
        'max_frames': 50, # Limit video length for memory
        'hidden_dim': 256,
        'drop': 0.3,
        'patience': 5,
        'grad_clip': 1.0,
        'class_weight': True,
        'use_commonsense': True,
        'commonsense_fusion': 'attention'
    }


def main():
    parser = argparse.ArgumentParser(description='MELD-FAIR + COSMIC + GraphSmile Main Training Script')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--textf_mode', type=str, default='concat4', 
                       choices=['concat4', 'concat2', 'sum4', 'sum2', 'single'],
                       help='Text feature fusion mode')
    parser.add_argument('--win', type=int, nargs=2, default=[3, 3], help='Window sizes [past, future]')
    parser.add_argument('--shift_win', type=int, default=3, help='Sentiment shift window')
    parser.add_argument('--heter_n_layers', type=int, nargs=3, default=[2, 2, 2], help='Heterogeneous layers')
    
    # COSMIC parameters
    parser.add_argument('--use_commonsense', action='store_true', default=True, help='Use commonsense features')
    parser.add_argument('--commonsense_fusion', type=str, default='attention', 
                       choices=['attention', 'concat', 'weighted'], help='Commonsense fusion method')
    
    # Video parameters
    parser.add_argument('--max_frames', type=int, default=50, help='Maximum frames per video')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--video_lr_factor', type=float, default=0.1, help='LR factor for video components')
    parser.add_argument('--l2', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--class_weight', action='store_true', default=True, help='Use class weights')
    parser.add_argument('--class_weight_mu', type=float, default=1.0, help='Class weight mu parameter')
    
    # Scheduler parameters
    parser.add_argument('--lr_factor', type=float, default=0.5, help='LR reduction factor')
    parser.add_argument('--lr_patience', type=int, default=2, help='LR scheduler patience')
    
    # System parameters
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=20, help='Log interval for training')
    
    # Data paths
    parser.add_argument('--meldfair_path', type=str, default='../MELD-FAIR',
                       help='Path to MELD-FAIR base folder')
    parser.add_argument('--cosmic_roberta_path', type=str, 
                       default='../features/meld_features_roberta.pkl')
    parser.add_argument('--cosmic_comet_path', type=str,
                       default='../features/meld_features_comet.pkl')
    
    # Model saving
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='meldfair_video_cosmic_graphsmile', help='Model name')
    
    # Modes
    parser.add_argument('--train_only', action='store_true', help='Only train, skip testing')
    parser.add_argument('--test_only', action='store_true', help='Only test, skip training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Print configuration
    print("🚀 MELD-FAIR + COSMIC + GraphSmile Training")
    print("=" * 50)
    print("📋 RECOMMENDED SETTINGS:")
    recommended = get_recommended_settings()
    for key, value in recommended.items():
        current_value = getattr(args, key, 'N/A')
        status = "✅" if current_value == value else "⚠️"
        print(f"   {status} {key}: {current_value} (recommended: {value})")
    print()
    
    # Initialize trainer
    trainer = EmotionRecognitionTrainer(args)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Initialize model
    trainer.initialize_model()
    
    train_results, dev_results, test_results = {}, {}, {}
    
    if not args.test_only:
        # Train the model
        best_f1, best_epoch = trainer.train()
        
        # Evaluate on training and dev sets
        train_results = trainer.comprehensive_evaluation(trainer.train_loader, 'train')
        dev_results = trainer.comprehensive_evaluation(trainer.dev_loader, 'dev')
        
        # Plot training history
        trainer.plot_training_history()
    
    if not args.train_only:
        # Load best model for testing
        if trainer.load_best_model():
            # Comprehensive test evaluation
            test_results = trainer.comprehensive_evaluation(trainer.test_loader, 'test')
        else:
            print("⚠️  Could not load best model for testing")
    
    # Save all results
    if train_results or dev_results or test_results:
        trainer.save_results(train_results, dev_results, test_results)
    
    print("\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved in: {args.save_dir}")


if __name__ == '__main__':
    main()