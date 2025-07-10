import torch
import torch.nn as nn
import torch.nn.functional as F
from module import HeterGConv_Edge, HeterGConvLayer
import numpy as np


class MELDFAIRCommonsenseGraphSmile(nn.Module):
    """
    MELD-FAIR compatible GraphSmile model that incorporates COSMIC's commonsense features
    for enhanced multimodal emotion recognition using realigned audio-visual data.
    """

    def __init__(self, args, n_classes_emo=7, n_classes_sent=3):
        super(MELDFAIRCommonsenseGraphSmile, self).__init__()
        
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.use_commonsense = getattr(args, 'use_commonsense', True)
        self.commonsense_fusion = getattr(args, 'commonsense_fusion', 'attention')
        self.textf_mode = getattr(args, 'textf_mode', 'concat4')
        
        # Dimension parameters
        self.hidden_dim = args.hidden_dim
        self.drop = args.drop
        
        # Input dimensions
        self.text_dim = 1024  # RoBERTa features
        self.audio_dim = 13   # MFCC features
        self.face_dim = 512   # Face CNN output dimension
        self.commonsense_dim = 768  # COMET features
        
        # Text feature processing (4 RoBERTa layers)
        self.text_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.text_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop)
            ) for _ in range(4)
        ])
        
        # Text fusion based on mode
        if self.textf_mode == 'concat4':
            self.text_fusion = nn.Sequential(
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop)
            )
        elif self.textf_mode == 'concat2':
            self.text_fusion = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.drop)
            )
        else:  # average or single layer
            self.text_fusion = nn.Identity()
        
        # Audio processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop)
        )
        
        # Face/Visual processing with CNN backbone
        self.face_cnn = FaceCNN(output_dim=self.face_dim)
        self.face_encoder = nn.Sequential(
            nn.Linear(self.face_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop)
        )
        
        # Commonsense feature processing
        if self.use_commonsense:
            self.commonsense_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.commonsense_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.drop)
                ) for _ in range(9)  # 9 COMET features
            ])
            
            # Commonsense fusion
            if self.commonsense_fusion == 'attention':
                self.commonsense_attention = CommonsenseAttention(self.hidden_dim, 9)
            elif self.commonsense_fusion == 'concat':
                self.commonsense_concat = nn.Sequential(
                    nn.Linear(9 * self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.drop)
                )
            elif self.commonsense_fusion == 'weighted':
                self.commonsense_weights = nn.Parameter(torch.ones(9) / 9)
        
        # Heterogeneous Graph Convolution layers
        # Text-Audio
        hetergconv_ta = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            self.hidden_dim, hetergconv_ta, args.heter_n_layers[0], self.drop, self.no_cuda
        )
        
        # Text-Visual
        hetergconv_tv = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            self.hidden_dim, hetergconv_tv, args.heter_n_layers[1], self.drop, self.no_cuda
        )
        
        # Audio-Visual
        hetergconv_av = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
        self.hetergconv_av = HeterGConv_Edge(
            self.hidden_dim, hetergconv_av, args.heter_n_layers[2], self.drop, self.no_cuda
        )
        
        # Commonsense-aware heterogeneous graph convolution
        if self.use_commonsense:
            hetergconv_tc = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
            self.hetergconv_tc = HeterGConv_Edge(
                self.hidden_dim, hetergconv_tc, args.heter_n_layers[0], self.drop, self.no_cuda
            )
            hetergconv_ac = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
            self.hetergconv_ac = HeterGConv_Edge(
                self.hidden_dim, hetergconv_ac, args.heter_n_layers[1], self.drop, self.no_cuda
            )
            hetergconv_vc = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
            self.hetergconv_vc = HeterGConv_Edge(
                self.hidden_dim, hetergconv_vc, args.heter_n_layers[2], self.drop, self.no_cuda
            )
        
        # Final fusion and classification
        num_modalities = 6 if self.use_commonsense else 3
        self.modal_fusion = nn.Sequential(
            nn.Linear(num_modalities * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop)
        )
        
        # Output layers
        self.emotion_classifier = nn.Linear(self.hidden_dim, n_classes_emo)
        self.sentiment_classifier = nn.Linear(self.hidden_dim, n_classes_sent)
        
    def forward(self, batch):
        """
        Forward pass for MELD-FAIR data format
        
        Args:
            batch: Dictionary containing:
                - audio: [batch, time, mfcc_features]
                - face_sequences: [batch, frames, channels, height, width]  
                - text_features: List of 4 [batch, text_dim] tensors
                - commonsense_features: List of 9 [batch, cs_dim] tensors (optional)
                - audio_mask: [batch, time] for masking padded audio
        """
        batch_size = batch['audio'].shape[0]
        
        # Process text features
        text_encoded = []
        for i, text_layer in enumerate(self.text_layers):
            text_feat = text_layer(batch['text_features'][i])
            text_encoded.append(text_feat)
        
        # Fuse text features
        if self.textf_mode == 'concat4':
            text_fused = self.text_fusion(torch.cat(text_encoded, dim=-1))
        elif self.textf_mode == 'concat2':
            text_fused = self.text_fusion(torch.cat(text_encoded[:2], dim=-1))
        elif self.textf_mode == 'sum4':
            text_fused = sum(text_encoded) / 4
        elif self.textf_mode == 'sum2':
            text_fused = (text_encoded[0] + text_encoded[1]) / 2
        else:  # Use first layer
            text_fused = text_encoded[0]
        
        # Process audio features
        audio_features = batch['audio']  # [batch, time, mfcc]
        audio_mask = batch['audio_mask']  # [batch, time]
        
        # Global average pooling for audio (considering mask)
        masked_audio = audio_features * audio_mask.unsqueeze(-1)
        audio_lengths = audio_mask.sum(dim=1, keepdim=True)
        audio_pooled = masked_audio.sum(dim=1) / (audio_lengths + 1e-8)
        audio_encoded = self.audio_encoder(audio_pooled)
        
        # Process face sequences
        face_sequences = batch['face_sequences']  # [batch, frames, channels, H, W]
        batch_size, num_frames = face_sequences.shape[:2]
        
        # Reshape for CNN processing
        face_reshaped = face_sequences.view(-1, *face_sequences.shape[2:])
        face_features = self.face_cnn(face_reshaped)  # [batch*frames, face_dim]
        face_features = face_features.view(batch_size, num_frames, -1)
        
        # Global average pooling for faces
        face_pooled = face_features.mean(dim=1)  # [batch, face_dim]
        face_encoded = self.face_encoder(face_pooled)
        
        # Process commonsense features
        if self.use_commonsense:
            commonsense_encoded = []
            for i, cs_layer in enumerate(self.commonsense_layers):
                cs_feat = cs_layer(batch['commonsense_features'][i])
                commonsense_encoded.append(cs_feat)
            
            # Fuse commonsense features
            if self.commonsense_fusion == 'attention':
                commonsense_fused = self.commonsense_attention(commonsense_encoded)
            elif self.commonsense_fusion == 'concat':
                commonsense_fused = self.commonsense_concat(torch.cat(commonsense_encoded, dim=-1))
            elif self.commonsense_fusion == 'weighted':
                weights = F.softmax(self.commonsense_weights, dim=0)
                commonsense_fused = sum(w * feat for w, feat in zip(weights, commonsense_encoded))
            else:  # average
                commonsense_fused = sum(commonsense_encoded) / len(commonsense_encoded)
        
        # Prepare for graph convolution (convert to sequence format)
        # For simplicity, we treat each sample as a sequence of length 1
        text_seq = text_fused.unsqueeze(0)  # [1, batch, hidden]
        audio_seq = audio_encoded.unsqueeze(0)  # [1, batch, hidden]  
        face_seq = face_encoded.unsqueeze(0)  # [1, batch, hidden]
        
        # Create dummy dialogue lengths (each sample is length 1)
        dia_lengths = [1] * batch_size
        
        # Convert to graph format
        text_graph = text_seq.squeeze(0)  # [batch, hidden]
        audio_graph = audio_seq.squeeze(0)  # [batch, hidden]
        face_graph = face_seq.squeeze(0)  # [batch, hidden]
        
        # Heterogeneous graph convolution
        # Text-Audio
        feat_ta, edge_index = self.hetergconv_ta(
            (text_graph, audio_graph), dia_lengths, self.win_p, self.win_f
        )
        
        # Text-Visual  
        feat_tv, edge_index = self.hetergconv_tv(
            (text_graph, face_graph), dia_lengths, self.win_p, self.win_f, edge_index
        )
        
        # Audio-Visual
        feat_av, edge_index = self.hetergconv_av(
            (audio_graph, face_graph), dia_lengths, self.win_p, self.win_f, edge_index
        )
        
        # Collect features for fusion
        fusion_features = [feat_ta[0], feat_ta[1], feat_tv[0], feat_tv[1], feat_av[0], feat_av[1]]
        
        # Commonsense-aware graph convolution
        if self.use_commonsense:
            commonsense_seq = commonsense_fused.unsqueeze(0)  # [1, batch, hidden]
            commonsense_graph = commonsense_seq.squeeze(0)  # [batch, hidden]
            
            # Text-Commonsense
            feat_tc, edge_index = self.hetergconv_tc(
                (text_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
            
            # Audio-Commonsense
            feat_ac, edge_index = self.hetergconv_ac(
                (audio_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
            
            # Visual-Commonsense
            feat_vc, edge_index = self.hetergconv_vc(
                (face_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
            
            # Add commonsense features to fusion
            fusion_features.extend([feat_tc[1], feat_ac[1], feat_vc[1]])  # commonsense node features
        
        # Concatenate all features for fusion
        fused_features = torch.cat(fusion_features, dim=-1)
        fused_output = self.modal_fusion(fused_features)
        
        # Final classification
        emotion_logits = self.emotion_classifier(fused_output)
        sentiment_logits = self.sentiment_classifier(fused_output)
        
        return emotion_logits, sentiment_logits, fused_output


class FaceCNN(nn.Module):
    """Simple CNN for face feature extraction"""
    
    def __init__(self, output_dim=512):
        super(FaceCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class CommonsenseAttention(nn.Module):
    """Attention mechanism for fusing multiple commonsense features"""
    
    def __init__(self, hidden_dim, num_features):
        super(CommonsenseAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, commonsense_features):
        """
        Args:
            commonsense_features: List of [batch, hidden_dim] tensors
        Returns:
            Fused commonsense feature: [batch, hidden_dim]
        """
        # Stack features: [batch, num_features, hidden_dim]
        stacked = torch.stack(commonsense_features, dim=1)
        
        # Compute attention weights: [batch, num_features, 1]
        attention_weights = self.attention(stacked)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention: [batch, hidden_dim]
        fused = (stacked * attention_weights).sum(dim=1)
        
        return fused