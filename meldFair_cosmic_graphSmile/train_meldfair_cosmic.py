import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

from meldfair_cosmic_model import MELDFAIRCommonsenseGraphSmile
from meldfair_cosmic_dataloader import get_MELDFAIR_COSMIC_loaders


def create_class_weight(mu=1):
    """Create class weights for MELD emotion labels"""
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = np.log(mu * total / labels_dict[key])
        weights.append(score)
    return weights


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, args=None):
    """Train or evaluate the model"""
    losses, preds, labels = [], [], []
    
    if train:
        model.train()
    else:
        model.eval()
    
    for i, batch in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        # Move batch to device
        if torch.cuda.is_available() and not args.no_cuda:
            for key in ['audio', 'face_sequences', 'emotion_labels', 'sentiment_labels', 'audio_mask']:
                if key in batch:
                    batch[key] = batch[key].cuda()
            
            # Move text and commonsense features
            batch['text_features'] = [tf.cuda() for tf in batch['text_features']]
            batch['commonsense_features'] = [cf.cuda() for cf in batch['commonsense_features']]
        
        # Forward pass
        try:
            emotion_logits, sentiment_logits, fused_features = model(batch)
            
            # Calculate loss (focusing on emotion recognition)
            emotion_labels = batch['emotion_labels']
            loss = loss_function(emotion_logits, emotion_labels)
            
            # Predictions
            pred = torch.argmax(emotion_logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(emotion_labels.cpu().numpy())
            losses.append(loss.item())
            
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if i % 50 == 0:
                print(f'Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}')
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    if preds:
        avg_loss = np.mean(losses)
        avg_accuracy = accuracy_score(labels, preds) * 100
        avg_fscore = f1_score(labels, preds, average='weighted') * 100
        
        return avg_loss, avg_accuracy, avg_fscore, labels, preds
    else:
        return float('nan'), float('nan'), float('nan'), [], []


def main():
    parser = argparse.ArgumentParser(description='MELD-FAIR + COSMIC + GraphSmile Training')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--textf_mode', type=str, default='concat4', 
                       choices=['concat4', 'concat2', 'sum4', 'sum2', 'single'],
                       help='Text feature fusion mode')
    parser.add_argument('--win', type=int, nargs=2, default=[3, 3], help='Window sizes [past, future]')
    parser.add_argument('--heter_n_layers', type=int, nargs=3, default=[2, 2, 2], help='Heterogeneous layers')
    
    # COSMIC parameters
    parser.add_argument('--use_commonsense', action='store_true', default=True, help='Use commonsense features')
    parser.add_argument('--commonsense_fusion', type=str, default='attention', 
                       choices=['attention', 'concat', 'weighted'], help='Commonsense fusion method')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--class_weight', action='store_true', help='Use class weights')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    # Data paths
    parser.add_argument('--meldfair_path', type=str, default='../MELD-FAIR',
                       help='Path to MELD-FAIR base folder')
    parser.add_argument('--cosmic_roberta_path', type=str, 
                       default='../conv-emotion/COSMIC/erc-training/meld/meld_features_roberta.pkl')
    parser.add_argument('--cosmic_comet_path', type=str,
                       default='../conv-emotion/COSMIC/erc-training/meld/meld_features_comet.pkl')
    
    # Model saving
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='meldfair_cosmic_graphsmile', help='Model name')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check CUDA
    if not args.no_cuda and torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        args.no_cuda = True
        device = torch.device('cpu')
    
    # Load data
    print('Loading MELD-FAIR + COSMIC data...')
    try:
        train_loader, dev_loader, test_loader = get_MELDFAIR_COSMIC_loaders(
            args.meldfair_path, args.cosmic_roberta_path, args.cosmic_comet_path,
            batch_size=args.batch_size, num_workers=0, pin_memory=not args.no_cuda
        )
        print(f'Data loaded successfully!')
        print(f'Train batches: {len(train_loader)}')
        print(f'Dev batches: {len(dev_loader)}')
        print(f'Test batches: {len(test_loader)}')
    except Exception as e:
        print(f'Error loading data: {e}')
        return
    
    # Initialize model
    print('Initializing model...')
    model = MELDFAIRCommonsenseGraphSmile(args, n_classes_emo=7, n_classes_sent=3)
    
    if not args.no_cuda:
        model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function
    if args.class_weight:
        loss_weights = torch.FloatTensor(create_class_weight())
        if not args.no_cuda:
            loss_weights = loss_weights.cuda()
        loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        loss_function = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print('Starting training...')
    best_fscore = 0
    best_epoch = 0
    patience_counter = 0
    
    train_losses, dev_losses = [], []
    train_fscores, dev_fscores = [], []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(
            model, loss_function, train_loader, epoch, optimizer, train=True, args=args
        )
        
        # Validation
        with torch.no_grad():
            dev_loss, dev_acc, dev_fscore, dev_labels, dev_preds = train_or_eval_model(
                model, loss_function, dev_loader, epoch, train=False, args=args
            )
        
        # Update learning rate
        scheduler.step(dev_fscore)
        
        # Record metrics
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        train_fscores.append(train_fscore)
        dev_fscores.append(dev_fscore)
        
        # Print results
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_fscore:.2f}%')
        print(f'  Dev   - Loss: {dev_loss:.4f}, Acc: {dev_acc:.2f}%, F1: {dev_fscore:.2f}%')
        print(f'  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if dev_fscore > best_fscore:
            best_fscore = dev_fscore
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(args.save_dir, f'{args.model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fscore': best_fscore,
                'args': args
            }, model_path)
            
            print(f'  🎉 New best F1: {best_fscore:.2f}% (saved to {model_path})')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{args.patience}')
        
        print('-' * 70)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'\nTraining completed. Best F1: {best_fscore:.2f}% at epoch {best_epoch+1}')
    
    # Load best model and test
    print('\nTesting best model...')
    checkpoint = torch.load(os.path.join(args.save_dir, f'{args.model_name}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        test_loss, test_acc, test_fscore, test_labels, test_preds = train_or_eval_model(
            model, loss_function, test_loader, 0, train=False, args=args
        )
    
    print(f'\n🏆 Final Test Results:')
    print(f'  Accuracy: {test_acc:.2f}%')
    print(f'  F1-Score: {test_fscore:.2f}%')
    print(f'  Loss: {test_loss:.4f}')
    
    # Classification report
    if len(test_labels) > 0:
        emotion_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
        print(f'\n📊 Detailed Classification Report:')
        print(classification_report(test_labels, test_preds, target_names=emotion_names, digits=3))
    
    # Save final results
    results = {
        'best_dev_fscore': best_fscore,
        'best_epoch': best_epoch,
        'test_accuracy': test_acc,
        'test_fscore': test_fscore,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'dev_losses': dev_losses,
        'train_fscores': train_fscores,
        'dev_fscores': dev_fscores
    }
    
    results_path = os.path.join(args.save_dir, f'{args.model_name}_results.txt')
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()