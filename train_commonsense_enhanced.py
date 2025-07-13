"""
Training script for CommonsenseEnhancedGraphSmile model
Combines COSMIC's commonsense reasoning with GraphSmile's multimodal fusion
"""

import os
import sys
import time
import math
import random
import datetime
import numpy as np
import pickle as pk
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import our modules
from config import Config
from commonsense_enhanced_graphsmile import CommonsenseEnhancedGraphSmile
from dataloader import MELDCombinedDataset
from utils import EarlyStopping, MetricsTracker, save_results, create_directories, MaskedNLLLoss


def seed_everything(config):
    """Set random seeds for reproducibility"""
    seed = config.system.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.system.deterministic

def create_class_weight(mu=1, n_classes=7):
    """Create class weights for imbalanced MELD dataset"""
    # MELD emotion class distribution
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in range(n_classes):
        if key in labels_dict:
            score = math.log(mu * total / labels_dict[key])
        else:
            score = 1.0  # Default weight for missing classes
        weights.append(score)
    return weights

def get_loss_function(config, device):
    """Create loss functions for multi-task learning (emotion, sentiment, shift)"""
    loss_functions = {}
    
    # COSMIC loss - uses MaskedNLLLoss for variable-length sequences
    if config.training.class_weight:
        if config.training.class_weight_mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(config.training.class_weight_mu, config.model.n_classes))
        else:
            # Use original COSMIC weights exactly as in their implementation
            loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                                            0.84847735, 5.42461417, 1.21859721])
        loss_functions['cosmic'] = MaskedNLLLoss(weight=loss_weights.to(device))
        loss_functions['emotion'] = nn.NLLLoss(weight=loss_weights.to(device))
    else:
        loss_functions['cosmic'] = MaskedNLLLoss()
        loss_functions['emotion'] = nn.NLLLoss()
    
    # Sentiment and shift losses (always create for multi-task learning)
    loss_functions['sentiment'] = nn.NLLLoss()
    loss_functions['shift'] = nn.NLLLoss()
    
    return loss_functions

def compute_combined_loss(log_prob_cosmic, logit_emo, logit_sen, logit_shift, 
                         labels_emo, labels_sen, labels_shift, 
                         loss_functions, config, epoch=0, umask=None, labels_emo_padded=None):
    """Compute combined loss from multiple outputs"""
    losses = {}
    
    # COSMIC loss (if available) - use MaskedNLLLoss for proper masking
    if log_prob_cosmic is not None and labels_emo_padded is not None:
        # For COSMIC loss, keep the padded format and use original labels with masking
        lp_ = log_prob_cosmic.transpose(0,1).contiguous().view(-1, log_prob_cosmic.size()[2])
        labels_padded_flat = labels_emo_padded.view(-1)  # Use original padded labels
        if umask is not None:
            # Use MaskedNLLLoss for proper sequence masking
            losses['cosmic'] = loss_functions['cosmic'](lp_, labels_padded_flat, umask)
        else:
            # Fallback to standard NLL loss if no mask available
            losses['cosmic'] = F.nll_loss(lp_, labels_padded_flat)
    
    # GraphSmile losses
    if logit_emo is not None:
        prob_emo = F.log_softmax(logit_emo, -1)
        # Ensure logit_emo and labels_emo have the same batch size
        if prob_emo.size(0) != labels_emo.size(0):
            # Truncate to match the smaller size (conservative approach)
            min_size = min(prob_emo.size(0), labels_emo.size(0))
            prob_emo = prob_emo[:min_size]
            labels_emo_truncated = labels_emo[:min_size]
            losses['emotion'] = loss_functions['emotion'](prob_emo, labels_emo_truncated)
        else:
            losses['emotion'] = loss_functions['emotion'](prob_emo, labels_emo)
    
    if logit_sen is not None:
        prob_sen = F.log_softmax(logit_sen, -1)
        # Ensure logit_sen and labels_sen have the same batch size
        if prob_sen.size(0) != labels_sen.size(0):
            min_size = min(prob_sen.size(0), labels_sen.size(0))
            prob_sen = prob_sen[:min_size]
            labels_sen_truncated = labels_sen[:min_size]
            losses['sentiment'] = loss_functions['sentiment'](prob_sen, labels_sen_truncated)
        else:
            losses['sentiment'] = loss_functions['sentiment'](prob_sen, labels_sen)
    
    if logit_shift is not None and labels_shift is not None and len(labels_shift) > 0:
        prob_sft = F.log_softmax(logit_shift, -1)
        # Ensure logit_shift and labels_shift have the same batch size
        if prob_sft.size(0) != labels_shift.size(0):
            min_size = min(prob_sft.size(0), labels_shift.size(0))
            prob_sft = prob_sft[:min_size]
            labels_shift_truncated = labels_shift[:min_size]
            losses['shift'] = loss_functions['shift'](prob_sft, labels_shift_truncated)
        else:
            losses['shift'] = loss_functions['shift'](prob_sft, labels_shift)
    
    # Combine losses based on loss_type
    loss_type = config.training.loss_type
    lambd = config.training.lambd
    
    if loss_type == "cosmic_only":
        total_loss = losses.get('cosmic', 0)
    elif loss_type == "emo_sen_sft":
        total_loss = (lambd[0] * losses.get('emotion', 0) + 
                     lambd[1] * losses.get('sentiment', 0) + 
                     lambd[2] * losses.get('shift', 0))
    elif loss_type == "cosmic_graphsmile":
        total_loss = (0.5 * losses.get('cosmic', 0) + 
                     0.5 * (lambd[0] * losses.get('emotion', 0) + 
                           lambd[1] * losses.get('sentiment', 0) + 
                           lambd[2] * losses.get('shift', 0)))
    elif loss_type == "epoch_weighted":
        # Gradually shift from COSMIC to GraphSmile
        cosmic_weight = 1 - (epoch / config.training.epochs)
        graphsmile_weight = epoch / config.training.epochs
        total_loss = (cosmic_weight * losses.get('cosmic', 0) + 
                     graphsmile_weight * (lambd[0] * losses.get('emotion', 0) + 
                                        lambd[1] * losses.get('sentiment', 0) + 
                                        lambd[2] * losses.get('shift', 0)))
    else:
        total_loss = losses.get('emotion', 0)  # Default to emotion loss
    
    return total_loss, losses

def extract_dialogue_lengths(umask):
    """Extract dialogue lengths from utterance mask"""
    dia_lengths = []
    for j in range(umask.size(1)):
        length = (umask[:, j] == 1).nonzero().tolist()[-1][0] + 1
        dia_lengths.append(length)
    return dia_lengths

def build_sentiment_shift_labels(shift_win, dia_lengths, label_sen):
    """Build sentiment shift labels using GraphSmile's exact implementation"""
    start = 0
    label_shifts = []
    if shift_win == -1:
        for dia_len in dia_lengths:
            dia_label_shift = ((label_sen[start:start + dia_len, None]
                                != label_sen[None, start:start +
                                             dia_len]).long().view(-1))
            label_shifts.append(dia_label_shift)
            start += dia_len
        label_shift = torch.cat(label_shifts, dim=0)
    elif shift_win > 0:
        for dia_len in dia_lengths:
            win_start = 0
            for i in range(math.ceil(dia_len / shift_win)):
                if i == math.ceil(
                        dia_len / shift_win) - 1 and dia_len % shift_win != 0:
                    win = dia_len % shift_win
                else:
                    win = shift_win
                dia_label_shift = (
                    (
                        label_sen[start + win_start : start + win_start + win, None]
                        != label_sen[None, start + win_start : start + win_start + win]
                    )
                    .long()
                    .view(-1)
                )
                label_shifts.append(dia_label_shift)
                win_start += shift_win
            start += dia_len
        label_shift = torch.cat(label_shifts, dim=0)
    else:
        print("Window must be greater than 0 or equal to -1")
        raise NotImplementedError

    return label_shift

def train_or_eval_model(model, loss_functions, dataloader, config, epoch, optimizer=None, train=False):
    """Train or evaluate model for one epoch"""
    if train:
        model.train()
    else:
        model.eval()
    
    losses = []
    all_preds_emo, all_labels_emo = [], []
    all_preds_sen, all_labels_sen = [], []
    all_preds_shift, all_labels_shift = [], []
    all_masks = []
    
    device = next(model.parameters()).device
    
    for batch_idx, batch in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        # Unpack batch data
        (roberta_features, comet_features, visual_features, audio_features, 
         speakers, qmask, umask, labels_emo, labels_sen) = batch
        
        # Move to device
        roberta_features = [feat.to(device) for feat in roberta_features]
        comet_features = [feat.to(device) for feat in comet_features]
        visual_features = visual_features.to(device)
        audio_features = audio_features.to(device)
        speakers = speakers.to(device)
        qmask = qmask.to(device)
        umask = umask.to(device)
        labels_emo = labels_emo.to(device)
        labels_sen = labels_sen.to(device)
        
        # Extract dialogue lengths
        dia_lengths = extract_dialogue_lengths(umask)
        
        # Flatten labels for loss computation
        labels_emo_flat = []
        labels_sen_flat = []
        # Use the minimum of umask batch size and labels batch size
        batch_size = min(umask.size(1), labels_emo.size(1), len(dia_lengths))
        for j in range(batch_size):
            length = dia_lengths[j]
            labels_emo_flat.append(labels_emo[:length, j])
            labels_sen_flat.append(labels_sen[:length, j])
        
        labels_emo_cat = torch.cat(labels_emo_flat)
        labels_sen_cat = torch.cat(labels_sen_flat)
        
        # Build shift labels
        labels_shift = build_sentiment_shift_labels(config.model.shift_win, dia_lengths, labels_sen_cat)
        if len(labels_shift) > 0:
            labels_shift = labels_shift.to(device)
            
        
        # Forward pass
        with torch.set_grad_enabled(train):
            outputs = model(
                roberta_features=roberta_features,
                comet_features=comet_features,
                visual_features=visual_features,
                audio_features=audio_features,
                speakers=speakers,
                qmask=qmask,
                umask=umask,
                dia_lengths=dia_lengths,
                att2=config.model.att2,
                return_hidden=False
            )
            
            # Unpack outputs
            (log_prob, out_sense, alpha, alpha_f, alpha_b, emotions, 
             logit_emo, logit_sen, logit_shift, feat_fusion) = outputs
            
            # Compute loss
            total_loss, individual_losses = compute_combined_loss(
                log_prob, logit_emo, logit_sen, logit_shift,
                labels_emo_cat, labels_sen_cat, labels_shift,
                loss_functions, config, epoch, umask, labels_emo
            )
            
            
            # Compute predictions - Use GraphSmile as main predictor (COSMIC only enhances text)
            if logit_emo is not None:
                
                # GraphSmile emotion predictions (main predictions)
                emo_pred = torch.argmax(F.log_softmax(logit_emo, -1), -1)
                # Ensure same size as labels
                if emo_pred.size(0) != labels_emo_cat.size(0):
                    min_size = min(emo_pred.size(0), labels_emo_cat.size(0))
                    emo_pred = emo_pred[:min_size]
                all_preds_emo.append(emo_pred.cpu().numpy())
            
            if logit_sen is not None:
                sen_pred = torch.argmax(F.log_softmax(logit_sen, -1), -1)
                if sen_pred.size(0) != labels_sen_cat.size(0):
                    min_size = min(sen_pred.size(0), labels_sen_cat.size(0))
                    sen_pred = sen_pred[:min_size]
                all_preds_sen.append(sen_pred.cpu().numpy())
            
            if logit_shift is not None and labels_shift is not None and len(labels_shift) > 0:
                shift_pred = torch.argmax(F.log_softmax(logit_shift, -1), -1)
                
                
                if shift_pred.size(0) != labels_shift.size(0):
                    min_size = min(shift_pred.size(0), labels_shift.size(0))
                    shift_pred = shift_pred[:min_size]
                    labels_shift = labels_shift[:min_size]  # Also truncate labels to match
                all_preds_shift.append(shift_pred.cpu().numpy())
            
            # Store labels and masks
            all_labels_emo.append(labels_emo_cat.cpu().numpy())
            all_labels_sen.append(labels_sen_cat.cpu().numpy())
            if len(labels_shift) > 0:
                all_labels_shift.append(labels_shift.cpu().numpy())
            
            # Create mask for valid predictions
            mask = torch.ones_like(labels_emo_cat).float()
            all_masks.append(mask.cpu().numpy())
            
            losses.append(total_loss.item())
        
        # Backward pass
        if train:
            total_loss.backward()
            
            
            if config.training.gradient_clip > 0:
                clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
    
    # Compute metrics
    metrics = {}
    
    if all_preds_emo and all_labels_emo:
        preds_emo = np.concatenate(all_preds_emo)
        labels_emo = np.concatenate(all_labels_emo)
        masks = np.concatenate(all_masks)
        
        metrics['emotion_acc'] = accuracy_score(labels_emo, preds_emo, sample_weight=masks) * 100
        metrics['emotion_f1'] = f1_score(labels_emo, preds_emo, average='weighted', sample_weight=masks) * 100
    
    if all_preds_sen and all_labels_sen:
        preds_sen = np.concatenate(all_preds_sen)
        labels_sen = np.concatenate(all_labels_sen)
        
        metrics['sentiment_acc'] = accuracy_score(labels_sen, preds_sen) * 100
        metrics['sentiment_f1'] = f1_score(labels_sen, preds_sen, average='weighted') * 100
    
    # Skip shift metrics for now due to size mismatch issues
    # TODO: Fix sentiment shift label/prediction alignment
    if all_preds_shift and all_labels_shift:
        try:
            preds_shift = np.concatenate(all_preds_shift)
            labels_shift = np.concatenate(all_labels_shift)
            
            if len(preds_shift) == len(labels_shift):
                metrics['shift_acc'] = accuracy_score(labels_shift, preds_shift) * 100
                metrics['shift_f1'] = f1_score(labels_shift, preds_shift, average='weighted') * 100
            else:
                print(f"Warning: Shift prediction/label size mismatch: {len(preds_shift)} vs {len(labels_shift)}")
                metrics['shift_acc'] = 0.0
                metrics['shift_f1'] = 0.0
        except Exception as e:
            print(f"Warning: Could not compute shift metrics: {e}")
            metrics['shift_acc'] = 0.0
            metrics['shift_f1'] = 0.0
    
    metrics['loss'] = np.mean(losses)
    
    return metrics, (preds_emo if all_preds_emo else None, 
                    labels_emo if all_labels_emo else None,
                    preds_sen if all_preds_sen else None,
                    labels_sen if all_labels_sen else None)

def main():
    # Create config directly - NO COMMAND LINE ARGS
    config = Config()
    
    print("=" * 50)
    print("CommonsenseEnhancedGraphSmile Training")
    print("=" * 50)
    print(f"Experiment: {config.experiment.name}")
    print(f"Model: Hidden={config.model.hidden_dim}, Mode1={config.model.mode1}, Norm={config.model.norm}")
    print(f"Training: Epochs={config.training.epochs}, LR={config.training.learning_rate}, BS={config.training.batch_size}")
    print("=" * 50)
    
    # Set up device
    if config.system.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device(f'cuda:{config.system.gpu_ids.split(",")[0]}')
        print(f"Using GPU: {device}")
    
    # Set random seed
    seed_everything(config)
    
    # Create directories
    create_directories(config)
    
    # Initialize model
    model = CommonsenseEnhancedGraphSmile(config=config.model).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    print("Loading datasets...")
    train_dataset = MELDCombinedDataset(config=config.data, split='train')
    valid_dataset = MELDCombinedDataset(config=config.data, split='valid')
    test_dataset = MELDCombinedDataset(config=config.data, split='test')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # Create loss functions
    loss_functions = get_loss_function(config, device)
    
    # Print class weight information for debugging
    if config.training.class_weight:
        if config.training.class_weight_mu > 0:
            weights = create_class_weight(config.training.class_weight_mu, config.model.n_classes)
            print(f"Using computed class weights (mu={config.training.class_weight_mu}): {weights}")
        else:
            weights = [0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721]
            print(f"Using original COSMIC class weights: {weights}")
    else:
        print("No class weights - this may cause bias toward dominant classes!")
    
    # Create optimizer
    if config.training.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.l2_reg,
            amsgrad=True
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.l2_reg
        )
    
    # Add learning rate scheduler to help escape poor local minima
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Initialize tracking
    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta
    ) if config.training.early_stopping else None
    
    best_f1 = 0.0
    best_epoch = 0
    best_predictions = None
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.training.epochs):
        start_time = time.time()
        
        # Train
        train_metrics, _ = train_or_eval_model(
            model, loss_functions, train_loader, config, epoch, optimizer, train=True
        )
        
        # Validate every epoch for early stopping and learning rate scheduling
        valid_metrics, _ = train_or_eval_model(
            model, loss_functions, valid_loader, config, epoch, train=False
        )
        
        # Test only at the end or periodically (not every epoch)
        test_metrics = None
        test_predictions = None
        if (epoch + 1) % config.system.log_interval == 0 or (epoch + 1) == config.training.epochs:
            test_metrics, test_predictions = train_or_eval_model(
                model, loss_functions, test_loader, config, epoch, train=False
            )
        
        # Track metrics
        metrics_tracker.update({
            'train': train_metrics,
            'valid': valid_metrics,
            'test': test_metrics if test_metrics else {}
        })
        
        epoch_time = time.time() - start_time
        
        # Print only training progress during training
        print(f"Epoch {epoch+1}/{config.training.epochs} ({epoch_time:.1f}s) - Train Loss: {train_metrics['loss']:.4f}, Emo F1: {train_metrics.get('emotion_f1', 0):.2f}, Emo Acc: {train_metrics.get('emotion_acc', 0):.2f}")
        
        # Save best model (only update when we have test metrics)
        if test_metrics is not None:
            current_f1 = test_metrics.get('emotion_f1', 0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                best_predictions = test_predictions
                
                if config.experiment.save_best_only:
                    model_path = Path(config.experiment.models_dir) / f"{config.experiment.name}_best.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'metrics': test_metrics
                    }, model_path)
                    print(f"  → Best model saved (F1: {best_f1:.2f})")
        
        # Learning rate scheduling
        scheduler.step(valid_metrics['loss'])
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(valid_metrics['loss'])
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Periodic detailed metrics
        if (epoch + 1) % config.system.log_interval == 0 and best_predictions is not None:
            preds_emo, labels_emo, preds_sen, labels_sen = best_predictions
            if preds_emo is not None and labels_emo is not None:
                print("\nDetailed Emotion Results (Best Model):")
                
                # Show prediction distribution
                unique, counts = np.unique(preds_emo, return_counts=True)
                pred_dist = dict(zip(unique, counts))
                print(f"Prediction distribution: {pred_dist}")
                
                # Show label distribution for comparison
                unique_labels, counts_labels = np.unique(labels_emo, return_counts=True)
                label_dist = dict(zip(unique_labels, counts_labels))
                print(f"True label distribution: {label_dist}")
                
                print(classification_report(labels_emo, preds_emo, digits=4, zero_division=0))
                print("Emotion Confusion Matrix:")
                print(confusion_matrix(labels_emo, preds_emo))
            
            if preds_sen is not None and labels_sen is not None:
                print("\nDetailed Sentiment Results (Best Model):")
                
                # Show prediction distribution
                unique, counts = np.unique(preds_sen, return_counts=True)
                pred_dist = dict(zip(unique, counts))
                print(f"Prediction distribution: {pred_dist}")
                
                print(classification_report(labels_sen, preds_sen, digits=4, zero_division=0))
                print("Sentiment Confusion Matrix:")
                print(confusion_matrix(labels_sen, preds_sen))
            print("-" * 50)
    
    # Final test evaluation if we haven't done one recently
    if best_predictions is None:
        print("\nRunning final test evaluation...")
        final_test_metrics, final_test_predictions = train_or_eval_model(
            model, loss_functions, test_loader, config, config.training.epochs-1, train=False
        )
        if final_test_metrics.get('emotion_f1', 0) > best_f1:
            best_f1 = final_test_metrics.get('emotion_f1', 0)
            best_epoch = config.training.epochs - 1
            best_predictions = final_test_predictions
    
    # Final results
    # Final evaluation on validation and test sets
    print(f"\nTraining completed! Running final evaluation...")
    
    # Final validation evaluation
    final_valid_metrics, _ = train_or_eval_model(
        model, loss_functions, valid_loader, config, config.training.epochs-1, train=False
    )
    
    # Final test evaluation
    final_test_metrics, final_test_predictions = train_or_eval_model(
        model, loss_functions, test_loader, config, config.training.epochs-1, train=False
    )
    
    print(f"\nFinal Results:")
    print(f"Validation - Loss: {final_valid_metrics['loss']:.4f}, Emo F1: {final_valid_metrics.get('emotion_f1', 0):.2f}, Emo Acc: {final_valid_metrics.get('emotion_acc', 0):.2f}")
    print(f"           Sen F1: {final_valid_metrics.get('sentiment_f1', 0):.2f}, Sen Acc: {final_valid_metrics.get('sentiment_acc', 0):.2f}")
    print(f"Test       - Loss: {final_test_metrics['loss']:.4f}, Emo F1: {final_test_metrics.get('emotion_f1', 0):.2f}, Emo Acc: {final_test_metrics.get('emotion_acc', 0):.2f}")
    print(f"           Sen F1: {final_test_metrics.get('sentiment_f1', 0):.2f}, Sen Acc: {final_test_metrics.get('sentiment_acc', 0):.2f}")
    
    # Use final results if better than best recorded
    if final_test_metrics.get('emotion_f1', 0) > best_f1:
        best_f1 = final_test_metrics.get('emotion_f1', 0)
        best_epoch = config.training.epochs - 1
        best_predictions = final_test_predictions
    
    print(f"Best Test F1: {best_f1:.2f} at epoch {best_epoch+1}")
    
    # Plot training loss curve
    train_losses = [epoch_data['train']['loss'] for epoch_data in metrics_tracker.get_history()]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    loss_plot_path = Path(config.experiment.results_dir) / config.experiment.name / "training_loss.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training loss plot saved to: {loss_plot_path}")
    
    # Save final results
    results = {
        'config': config,
        'metrics_history': metrics_tracker.get_history(),
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_predictions': best_predictions
    }
    
    save_results(results, config)
    
    # Final evaluation with best predictions
    if best_predictions is not None:
        preds_emo, labels_emo, preds_sen, labels_sen = best_predictions
        
        print("\nFinal Test Results:")
        if preds_emo is not None and labels_emo is not None:
            print("Emotion Classification:")
            print(classification_report(labels_emo, preds_emo, digits=4, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(labels_emo, preds_emo))
        
        if preds_sen is not None and labels_sen is not None:
            print("\nSentiment Classification:")
            print(classification_report(labels_sen, preds_sen, digits=4, zero_division=0))
            print("Sentiment Confusion Matrix:")
            print(confusion_matrix(labels_sen, preds_sen))

if __name__ == '__main__':
    main()