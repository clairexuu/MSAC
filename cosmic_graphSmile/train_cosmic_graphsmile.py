import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
import pickle

from cosmic_graphsmile_model import CommonsenseGraphSmile
from cosmic_dataloader import get_MELD_COSMIC_loaders
from utils import batch_to_all_labels


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
    
    for i, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        # Unpack data
        (text0, text1, text2, text3, visual, audio, speakers, umask,
         x_intent, x_attr, x_need, x_want, x_effect, x_react, o_want, o_effect, o_react,
         emotion_labels, sentiment_labels, vid) = data
        
        # Move to GPU if available
        if torch.cuda.is_available() and not args.no_cuda:
            text0, text1, text2, text3 = text0.cuda(), text1.cuda(), text2.cuda(), text3.cuda()
            visual, audio, speakers, umask = visual.cuda(), audio.cuda(), speakers.cuda(), umask.cuda()
            x_intent, x_attr, x_need, x_want = x_intent.cuda(), x_attr.cuda(), x_need.cuda(), x_want.cuda()
            x_effect, x_react, o_want, o_effect, o_react = x_effect.cuda(), x_react.cuda(), o_want.cuda(), o_effect.cuda(), o_react.cuda()
            emotion_labels, sentiment_labels = emotion_labels.cuda(), sentiment_labels.cuda()
        
        # Get dialogue lengths
        dia_lengths = [torch.sum(umask[i]).int().item() for i in range(umask.size(0))]
        
        # Forward pass
        logit_emo, logit_sen, logit_shift, feat_fusion = model(
            text0, text1, text2, text3, visual, audio, umask, speakers, dia_lengths,
            x_intent, x_attr, x_need, x_want, x_effect, x_react, o_want, o_effect, o_react
        )
        
        # Convert to batch format for loss calculation
        flat_emotion_labels = batch_to_all_labels(emotion_labels, dia_lengths)
        flat_logit_emo = batch_to_all_labels(logit_emo, dia_lengths)
        flat_umask = batch_to_all_labels(umask, dia_lengths)
        
        # Calculate loss
        loss = loss_function(flat_logit_emo, flat_emotion_labels, flat_umask)
        
        # Predictions
        pred = torch.argmax(flat_logit_emo, dim=1)
        preds.append(pred.data.cpu().numpy())
        labels.append(flat_emotion_labels.data.cpu().numpy())
        losses.append(loss.item() * flat_umask.sum().item())
        
        if train:
            loss.backward()
            optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}')
    
    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        # Note: Need to implement mask handling for proper evaluation
        
        avg_loss = np.sum(losses) / len(preds)
        avg_accuracy = accuracy_score(labels, preds) * 100
        avg_fscore = f1_score(labels, preds, average='weighted') * 100
        
        return avg_loss, avg_accuracy, avg_fscore, labels, preds
    else:
        return float('nan'), float('nan'), float('nan'), [], []


class MaskedNLLLoss(nn.Module):
    """Masked NLL Loss for sequence data"""
    
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')
    
    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        mask_ = mask.view(-1, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def main():
    parser = argparse.ArgumentParser(description='COSMIC + GraphSmile Training')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--textf_mode', type=str, default='concat4', help='Text feature mode')
    parser.add_argument('--win', type=int, nargs=2, default=[5, 5], help='Window sizes [past, future]')
    parser.add_argument('--modals', type=str, default='tva', help='Modalities to use')
    parser.add_argument('--shift_win', type=int, default=3, help='Shift window size')
    parser.add_argument('--heter_n_layers', type=int, nargs=3, default=[2, 2, 2], help='Heterogeneous layers')
    
    # COSMIC parameters
    parser.add_argument('--use_commonsense', type=bool, default=True, help='Use commonsense features')
    parser.add_argument('--commonsense_fusion', type=str, default='attention', 
                       choices=['attention', 'concat', 'weighted'], help='Commonsense fusion method')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--class_weight', action='store_true', help='Use class weights')
    
    # Data paths
    parser.add_argument('--graphsmile_path', type=str, default='datasets/meld_multi_features.pkl')
    parser.add_argument('--cosmic_roberta_path', type=str, 
                       default='../conv-emotion/COSMIC/erc-training/meld/meld_features_roberta.pkl')
    parser.add_argument('--cosmic_comet_path', type=str,
                       default='../conv-emotion/COSMIC/erc-training/meld/meld_features_comet.pkl')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check CUDA
    if not args.no_cuda and torch.cuda.is_available():
        print('Using CUDA')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')
        args.no_cuda = True
    
    # Load data
    print('Loading data...')
    train_loader, test_loader = get_MELD_COSMIC_loaders(
        args.graphsmile_path, args.cosmic_roberta_path, args.cosmic_comet_path,
        batch_size=args.batch_size, num_workers=0, pin_memory=not args.no_cuda
    )
    
    # Model parameters
    embedding_dims = [1024, 342, 300]  # Text, Visual, Audio dimensions
    n_classes_emo = 7  # MELD emotion classes
    
    # Initialize model
    model = CommonsenseGraphSmile(args, embedding_dims, n_classes_emo, commonsense_dim=768)
    
    if not args.no_cuda:
        model = model.cuda()
    
    # Loss function
    if args.class_weight:
        loss_weights = torch.FloatTensor(create_class_weight())
        if not args.no_cuda:
            loss_weights = loss_weights.cuda()
        loss_function = MaskedNLLLoss(loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    # Training loop
    print('Starting training...')
    best_fscore = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(
            model, loss_function, train_loader, epoch, optimizer, train=True, args=args
        )
        
        # Evaluation
        test_loss, test_acc, test_fscore, test_labels, test_preds = train_or_eval_model(
            model, loss_function, test_loader, epoch, train=False, args=args
        )
        
        # Print results
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_fscore:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_fscore:.2f}%')
        print(f'  Time: {time.time() - start_time:.2f}s')
        
        # Save best model
        if test_fscore > best_fscore:
            best_fscore = test_fscore
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_cosmic_graphsmile_model.pth')
            print(f'  New best F1: {best_fscore:.2f}%')
        
        print('-' * 50)
    
    print(f'Training completed. Best F1: {best_fscore:.2f}% at epoch {best_epoch+1}')
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_cosmic_graphsmile_model.pth'))
    test_loss, test_acc, test_fscore, test_labels, test_preds = train_or_eval_model(
        model, loss_function, test_loader, 0, train=False, args=args
    )
    
    print(f'Final Test Results:')
    print(f'  Accuracy: {test_acc:.2f}%')
    print(f'  F1-Score: {test_fscore:.2f}%')
    print(f'  Loss: {test_loss:.4f}')
    
    # Classification report
    if len(test_labels) > 0:
        print('\nClassification Report:')
        print(classification_report(test_labels, test_preds))


if __name__ == '__main__':
    main()