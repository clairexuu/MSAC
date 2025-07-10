import torch
import torch.nn as nn
import torch.nn.functional as F
from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat
from utils import batch_to_all_tva
import numpy as np


class MELDFAIRVideoCommonsenseGraphSmile(nn.Module):
    """
    MELD-FAIR Video + COSMIC + GraphSmile model for enhanced multimodal emotion recognition.
    Uses full video sequences with temporal modeling and heterogeneous graph convolution.
    """

    def __init__(self, args, n_classes_emo=7, n_classes_sent=3):
        super(MELDFAIRVideoCommonsenseGraphSmile, self).__init__()
        
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.use_commonsense = getattr(args, 'use_commonsense', True)
        self.commonsense_fusion = getattr(args, 'commonsense_fusion', 'attention')
        self.textf_mode = getattr(args, 'textf_mode', 'concat4')
        self.shift_win = getattr(args, 'shift_win', 3)
        
        # Dimension parameters
        self.hidden_dim = args.hidden_dim
        self.drop = args.drop
        
        # Input dimensions
        self.text_dim = 1024  # RoBERTa features
        self.audio_dim = 13   # MFCC features
        self.video_dim = 512  # Video CNN output dimension
        self.commonsense_dim = 768  # COMET features
        
        # Text feature processing (4 RoBERTa layers)
        self.text_batchnorms = nn.ModuleList([
            nn.BatchNorm1d(self.text_dim) for _ in range(4)
        ])
        
        self.text_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.text_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.drop)
            ) for _ in range(4)
        ])
        
        # Text fusion based on mode
        if self.textf_mode == 'concat4':
            text_fusion_dim = 4 * self.hidden_dim
        elif self.textf_mode == 'concat2':
            text_fusion_dim = 2 * self.hidden_dim
        else:
            text_fusion_dim = self.hidden_dim
            
        self.text_fusion = nn.Sequential(
            nn.Linear(text_fusion_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop)
        ) if text_fusion_dim != self.hidden_dim else nn.Identity()
        
        # Audio processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop)
        )
        
        # Video processing with temporal modeling (inspired by MELD-FAIR)
        self.video_frontend = VideoFrontend(output_dim=self.video_dim)
        self.video_temporal = VideoTemporalEncoder(
            input_dim=self.video_dim, 
            hidden_dim=self.hidden_dim,
            dropout=self.drop
        )
        
        # Commonsense feature processing
        if self.use_commonsense:
            self.commonsense_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.commonsense_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(self.drop)
                ) for _ in range(9)  # 9 COMET features
            ])
            
            # Commonsense fusion
            if self.commonsense_fusion == 'attention':
                self.commonsense_attention = CommonsenseAttention(self.hidden_dim, 9)
            elif self.commonsense_fusion == 'concat':
                self.commonsense_concat = nn.Sequential(
                    nn.Linear(9 * self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(self.drop)
                )
            elif self.commonsense_fusion == 'weighted':
                self.commonsense_weights = nn.Parameter(torch.ones(9) / 9)
        
        # GraphSmile Heterogeneous Graph Convolution layers
        # Text-Audio
        hetergconv_ta = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            self.hidden_dim, hetergconv_ta, args.heter_n_layers[0], self.drop, self.no_cuda
        )
        
        # Text-Video
        hetergconv_tv = HeterGConvLayer(self.hidden_dim, self.drop, self.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            self.hidden_dim, hetergconv_tv, args.heter_n_layers[1], self.drop, self.no_cuda
        )
        
        # Audio-Video
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
        
        # Final fusion following GraphSmile pattern
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        
        # Output layers
        self.emo_output = nn.Linear(self.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(self.hidden_dim, n_classes_sent)
        
        # Sentiment shifting feature (from GraphSmile)
        self.senshift = SenShift_Feat(self.hidden_dim, self.drop, self.shift_win)
        
    def forward(self, batch):
        """
        Forward pass for MELD-FAIR Video data format
        
        Args:
            batch: Dictionary containing:
                - audio: [batch, time, mfcc_features]
                - video_sequences: [batch, frames, channels, height, width]  
                - text_features: List of 4 [batch, text_dim] tensors
                - commonsense_features: List of 9 [batch, cs_dim] tensors (optional)
                - audio_mask: [batch, time] for masking padded audio
                - video_mask: [batch, frames] for masking padded video
        """
        batch_size = batch['audio'].shape[0]
        
        # Process text features with batch normalization (GraphSmile style)
        text_encoded = []
        for i, (text_layer, batchnorm) in enumerate(zip(self.text_layers, self.text_batchnorms)):
            # Apply batch normalization
            text_feat = batch['text_features'][i]  # [batch, text_dim]
            text_feat = batchnorm(text_feat)
            text_feat = text_layer(text_feat)
            text_encoded.append(text_feat)
        
        # Fuse text features (GraphSmile style)
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
        
        # Process video sequences with temporal modeling
        video_sequences = batch['video_sequences']  # [batch, frames, channels, H, W]
        video_mask = batch['video_mask']  # [batch, frames]
        
        # Extract spatial features for each frame
        batch_size, max_frames = video_sequences.shape[:2]
        video_reshaped = video_sequences.view(-1, *video_sequences.shape[2:])  # [batch*frames, C, H, W]
        video_spatial_features = self.video_frontend(video_reshaped)  # [batch*frames, video_dim]
        video_spatial_features = video_spatial_features.view(batch_size, max_frames, -1)  # [batch, frames, video_dim]
        
        # Apply temporal encoding
        video_encoded = self.video_temporal(video_spatial_features, video_mask)  # [batch, hidden_dim]
        
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
        
        # Prepare for GraphSmile heterogeneous graph convolution
        # Convert to sequence format (each utterance as a single time step)
        text_seq = text_fused.unsqueeze(0)  # [1, batch, hidden]
        audio_seq = audio_encoded.unsqueeze(0)  # [1, batch, hidden]  
        video_seq = video_encoded.unsqueeze(0)  # [1, batch, hidden]
        
        # Create dialogue lengths (each sample is length 1)
        dia_lengths = [1] * batch_size
        
        # Apply batch_to_all_tva conversion (GraphSmile format)
        text_graph, audio_graph, video_graph = batch_to_all_tva(
            text_seq, audio_seq, video_seq, dia_lengths, self.no_cuda
        )
        
        # Heterogeneous graph convolution (GraphSmile pipeline)
        # Text-Audio
        feat_ta, edge_index = self.hetergconv_ta(
            (text_graph, audio_graph), dia_lengths, self.win_p, self.win_f
        )
        
        # Text-Video  
        feat_tv, edge_index = self.hetergconv_tv(
            (text_graph, video_graph), dia_lengths, self.win_p, self.win_f, edge_index
        )
        
        # Audio-Video
        feat_av, edge_index = self.hetergconv_av(
            (audio_graph, video_graph), dia_lengths, self.win_p, self.win_f, edge_index
        )
        
        # Commonsense-aware graph convolution
        if self.use_commonsense:
            commonsense_seq = commonsense_fused.unsqueeze(0)  # [1, batch, hidden]
            commonsense_graph = batch_to_all_tva(
                commonsense_seq, commonsense_seq, commonsense_seq, dia_lengths, self.no_cuda
            )[0]  # Use first output
            
            # Text-Commonsense
            feat_tc, edge_index = self.hetergconv_tc(
                (text_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
            
            # Audio-Commonsense
            feat_ac, edge_index = self.hetergconv_ac(
                (audio_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
            
            # Video-Commonsense
            feat_vc, edge_index = self.hetergconv_vc(
                (video_graph, commonsense_graph), dia_lengths, self.win_p, self.win_f, edge_index
            )
        
        # Modal fusion (GraphSmile style)
        if self.use_commonsense:
            feat_fusion = (self.modal_fusion(feat_ta[0]) + self.modal_fusion(feat_ta[1]) + 
                          self.modal_fusion(feat_tv[0]) + self.modal_fusion(feat_tv[1]) +
                          self.modal_fusion(feat_av[0]) + self.modal_fusion(feat_av[1]) +
                          self.modal_fusion(feat_tc[1]) + self.modal_fusion(feat_ac[1]) + 
                          self.modal_fusion(feat_vc[1])) / 9
        else:
            feat_fusion = (self.modal_fusion(feat_ta[0]) + self.modal_fusion(feat_ta[1]) + 
                          self.modal_fusion(feat_tv[0]) + self.modal_fusion(feat_tv[1]) +
                          self.modal_fusion(feat_av[0]) + self.modal_fusion(feat_av[1])) / 6
        
        # Final predictions
        logit_emo = self.emo_output(feat_fusion)
        logit_sen = self.sen_output(feat_fusion)
        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)
        
        return logit_emo, logit_sen, logit_shift, feat_fusion


class VideoFrontend(nn.Module):
    """Video frontend for spatial feature extraction from face frames"""
    
    def __init__(self, output_dim=512):
        super(VideoFrontend, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(512, output_dim)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(self._make_block(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(planes, planes))
        return nn.Sequential(*layers)
    
    def _make_block(self, inplanes, planes, stride=1):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output


class VideoTemporalEncoder(nn.Module):
    """Temporal encoder for video sequences using TCN"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(VideoTemporalEncoder, self).__init__()
        
        # Temporal Convolution Network (TCN)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention for temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, video_features, video_mask):
        """
        Args:
            video_features: [batch, frames, input_dim]
            video_mask: [batch, frames]
        Returns:
            Aggregated video features: [batch, hidden_dim]
        """
        batch_size, max_frames, input_dim = video_features.shape
        
        # Apply TCN: [batch, input_dim, frames] -> [batch, hidden_dim, frames]
        x = video_features.transpose(1, 2)  # [batch, input_dim, frames]
        x = self.tcn(x)  # [batch, hidden_dim, frames]
        x = x.transpose(1, 2)  # [batch, frames, hidden_dim]
        
        # Apply mask to zero out padded frames
        mask = video_mask.unsqueeze(-1)  # [batch, frames, 1]
        x = x * mask
        
        # Self-attention for temporal aggregation
        # Convert to [frames, batch, hidden_dim] for MultiheadAttention
        x_attn = x.transpose(0, 1)  # [frames, batch, hidden_dim]
        
        # Create attention mask (True means ignore)
        attn_mask = ~video_mask.bool()  # [batch, frames]
        
        # Apply self-attention
        x_attn, _ = self.temporal_attention(x_attn, x_attn, x_attn, key_padding_mask=attn_mask)
        x_attn = self.norm(x_attn)
        
        # Convert back to [batch, frames, hidden_dim]
        x = x_attn.transpose(0, 1)
        
        # Global average pooling with mask
        valid_lengths = video_mask.sum(dim=1, keepdim=True)  # [batch, 1]
        aggregated = (x * mask).sum(dim=1) / (valid_lengths + 1e-8)  # [batch, hidden_dim]
        
        return aggregated


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