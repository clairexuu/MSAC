from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import batch_to_all_tva
import numpy as np


class CommonsenseGraphSmile(nn.Module):
    """
    Enhanced GraphSmile model that incorporates COSMIC's commonsense features
    for improved multimodal emotion recognition.
    """

    def __init__(self, args, embedding_dims, n_classes_emo, commonsense_dim=768):
        super(CommonsenseGraphSmile, self).__init__()
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win
        self.use_commonsense = getattr(args, 'use_commonsense', True)
        self.commonsense_fusion = getattr(args, 'commonsense_fusion', 'attention')  # 'attention', 'concat', 'weighted'

        # Text feature processing (4 layers)
        self.batchnorms_t = nn.ModuleList(
            nn.BatchNorm1d(embedding_dims[0]) for _ in range(4))

        in_dims_t = (4 * embedding_dims[0] if args.textf_mode == "concat4" else
                     (2 * embedding_dims[0]
                      if args.textf_mode == "concat2" else embedding_dims[0]))
        self.dim_layer_t = nn.Sequential(nn.Linear(in_dims_t, args.hidden_dim),
                                         nn.LeakyReLU(), nn.Dropout(args.drop))
        
        # Visual and Audio processing
        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

        # Commonsense feature processing
        if self.use_commonsense:
            # Process each of the 9 commonsense features
            self.commonsense_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(commonsense_dim, args.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(args.drop),
                ) for _ in range(9)  # 9 COMET features
            ])
            
            # Commonsense fusion layer
            if self.commonsense_fusion == 'attention':
                self.commonsense_attention = CommonsenseAttention(args.hidden_dim, 9)
            elif self.commonsense_fusion == 'concat':
                self.commonsense_concat_layer = nn.Sequential(
                    nn.Linear(9 * args.hidden_dim, args.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(args.drop),
                )
            elif self.commonsense_fusion == 'weighted':
                self.commonsense_weights = nn.Parameter(torch.ones(9) / 9)

        # Heterogeneous Graph Convolution for original modalities
        hetergconvLayer_tv = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            args.hidden_dim, hetergconvLayer_tv, args.heter_n_layers[0], args.drop, args.no_cuda,
        )
        hetergconvLayer_ta = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            args.hidden_dim, hetergconvLayer_ta, args.heter_n_layers[1], args.drop, args.no_cuda,
        )
        hetergconvLayer_va = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
        self.hetergconv_va = HeterGConv_Edge(
            args.hidden_dim, hetergconvLayer_va, args.heter_n_layers[2], args.drop, args.no_cuda,
        )

        # Commonsense-aware heterogeneous graph convolution
        if self.use_commonsense:
            hetergconvLayer_tc = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
            self.hetergconv_tc = HeterGConv_Edge(
                args.hidden_dim, hetergconvLayer_tc, args.heter_n_layers[0], args.drop, args.no_cuda,
            )
            hetergconvLayer_vc = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
            self.hetergconv_vc = HeterGConv_Edge(
                args.hidden_dim, hetergconvLayer_vc, args.heter_n_layers[1], args.drop, args.no_cuda,
            )
            hetergconvLayer_ac = HeterGConvLayer(args.hidden_dim, args.drop, args.no_cuda)
            self.hetergconv_ac = HeterGConv_Edge(
                args.hidden_dim, hetergconvLayer_ac, args.heter_n_layers[2], args.drop, args.no_cuda,
            )

        # Modal fusion
        fusion_input_dim = args.hidden_dim * (7 if self.use_commonsense else 6)
        self.modal_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

        # Output layers
        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SenShift_Feat(args.hidden_dim, args.drop, args.shift_win)

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_v, feature_a, umask, qmask, dia_lengths,
                # COSMIC commonsense features
                x_intent=None, x_attr=None, x_need=None, x_want=None,
                x_effect=None, x_react=None, o_want=None, o_effect=None, o_react=None):

        # Process text features
        (
            (seq_len_t, batch_size_t, feature_dim_t),
            (seq_len_v, batch_size_v, feature_dim_v),
            (seq_len_a, batch_size_a, feature_dim_a),
        ) = [feature_t0.shape, feature_v.shape, feature_a.shape]
        
        features_t = [
            batchnorm_t(feature_t.transpose(0, 1).reshape(
                -1, feature_dim_t)).reshape(-1, seq_len_t,
                                            feature_dim_t).transpose(1, 0)
            for batchnorm_t, feature_t in
            zip(self.batchnorms_t,
                [feature_t0, feature_t1, feature_t2, feature_t3])
        ]
        feature_t0, feature_t1, feature_t2, feature_t3 = features_t

        # Text feature fusion
        dim_layer_dict_t = {
            "concat4": lambda: self.dim_layer_t(
                torch.cat([feature_t0, feature_t1, feature_t2, feature_t3], dim=-1)),
            "sum4": lambda: (self.dim_layer_t(feature_t0) + self.dim_layer_t(feature_t1) + 
                           self.dim_layer_t(feature_t2) + self.dim_layer_t(feature_t3)) / 4,
            "concat2": lambda: self.dim_layer_t(
                torch.cat([feature_t0, feature_t1], dim=-1)),
            "sum2": lambda: (self.dim_layer_t(feature_t0) + self.dim_layer_t(feature_t1)) / 2,
            "textf0": lambda: self.dim_layer_t(feature_t0),
            "textf1": lambda: self.dim_layer_t(feature_t1),
            "textf2": lambda: self.dim_layer_t(feature_t2),
            "textf3": lambda: self.dim_layer_t(feature_t3),
        }
        featdim_t = dim_layer_dict_t[self.textf_mode]()
        featdim_v, featdim_a = self.dim_layer_v(feature_v), self.dim_layer_a(feature_a)

        # Process commonsense features
        if self.use_commonsense and x_intent is not None:
            commonsense_features = [
                x_intent, x_attr, x_need, x_want, x_effect, x_react, o_want, o_effect, o_react
            ]
            
            # Transform each commonsense feature
            processed_commonsense = []
            for i, cs_feat in enumerate(commonsense_features):
                processed_cs = self.commonsense_layers[i](cs_feat)
                processed_commonsense.append(processed_cs)
            
            # Fuse commonsense features
            if self.commonsense_fusion == 'attention':
                featdim_c = self.commonsense_attention(processed_commonsense)
            elif self.commonsense_fusion == 'concat':
                featdim_c = self.commonsense_concat_layer(
                    torch.cat(processed_commonsense, dim=-1))
            elif self.commonsense_fusion == 'weighted':
                weights = F.softmax(self.commonsense_weights, dim=0)
                featdim_c = sum(w * feat for w, feat in zip(weights, processed_commonsense))
            else:  # default: simple average
                featdim_c = sum(processed_commonsense) / len(processed_commonsense)
        else:
            featdim_c = None

        # Convert to batch format for graph convolution
        emo_t, emo_v, emo_a = featdim_t, featdim_v, featdim_a
        emo_t, emo_v, emo_a = batch_to_all_tva(emo_t, emo_v, emo_a, dia_lengths, self.no_cuda)
        
        if featdim_c is not None:
            emo_c = featdim_c
            emo_c = batch_to_all_tva(emo_c, emo_c, emo_c, dia_lengths, self.no_cuda)[0]  # Use first output

        # Original heterogeneous graph convolution
        featheter_tv, heter_edge_index = self.hetergconv_tv(
            (emo_t, emo_v), dia_lengths, self.win_p, self.win_f)
        featheter_ta, heter_edge_index = self.hetergconv_ta(
            (emo_t, emo_a), dia_lengths, self.win_p, self.win_f, heter_edge_index)
        featheter_va, heter_edge_index = self.hetergconv_va(
            (emo_v, emo_a), dia_lengths, self.win_p, self.win_f, heter_edge_index)

        # Commonsense-aware graph convolution
        if self.use_commonsense and featdim_c is not None:
            featheter_tc, heter_edge_index = self.hetergconv_tc(
                (emo_t, emo_c), dia_lengths, self.win_p, self.win_f, heter_edge_index)
            featheter_vc, heter_edge_index = self.hetergconv_vc(
                (emo_v, emo_c), dia_lengths, self.win_p, self.win_f, heter_edge_index)
            featheter_ac, heter_edge_index = self.hetergconv_ac(
                (emo_a, emo_c), dia_lengths, self.win_p, self.win_f, heter_edge_index)

        # Modal fusion
        if self.use_commonsense and featdim_c is not None:
            feat_fusion = torch.cat([
                self.modal_fusion(featheter_tv[0]),
                self.modal_fusion(featheter_ta[0]),
                self.modal_fusion(featheter_tv[1]),
                self.modal_fusion(featheter_va[0]),
                self.modal_fusion(featheter_ta[1]),
                self.modal_fusion(featheter_va[1]),
                self.modal_fusion(featheter_tc[1]),  # commonsense features
            ], dim=-1)
            feat_fusion = self.modal_fusion(feat_fusion)
        else:
            feat_fusion = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(featheter_ta[0]) + 
                          self.modal_fusion(featheter_tv[1]) + self.modal_fusion(featheter_va[0]) + 
                          self.modal_fusion(featheter_ta[1]) + self.modal_fusion(featheter_va[1])) / 6

        # Output predictions
        logit_emo = self.emo_output(feat_fusion)
        logit_sen = self.sen_output(feat_fusion)
        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)

        return logit_emo, logit_sen, logit_shift, feat_fusion


class CommonsenseAttention(nn.Module):
    """
    Attention mechanism for fusing multiple commonsense features.
    """
    
    def __init__(self, hidden_dim, num_commonsense_features):
        super(CommonsenseAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_commonsense_features
        
        # Attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, commonsense_features):
        """
        Args:
            commonsense_features: List of [seq_len, batch, hidden_dim] tensors
        Returns:
            Fused commonsense feature: [seq_len, batch, hidden_dim]
        """
        # Stack features: [num_features, seq_len, batch, hidden_dim]
        stacked_features = torch.stack(commonsense_features, dim=0)
        
        # Compute attention scores
        # Reshape to [num_features * seq_len * batch, hidden_dim]
        num_features, seq_len, batch_size, hidden_dim = stacked_features.shape
        reshaped_features = stacked_features.view(-1, hidden_dim)
        
        # Compute attention scores
        attention_scores = self.attention_weights(reshaped_features)  # [num_features * seq_len * batch, 1]
        attention_scores = attention_scores.view(num_features, seq_len, batch_size)  # [num_features, seq_len, batch]
        
        # Apply softmax across features
        attention_weights = F.softmax(attention_scores, dim=0)  # [num_features, seq_len, batch]
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(-1)  # [num_features, seq_len, batch, 1]
        weighted_features = stacked_features * attention_weights  # [num_features, seq_len, batch, hidden_dim]
        
        # Sum across features
        fused_features = torch.sum(weighted_features, dim=0)  # [seq_len, batch, hidden_dim]
        
        return fused_features