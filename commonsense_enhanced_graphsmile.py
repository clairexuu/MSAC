import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
import copy
from itertools import permutations, product
from torch.nn import Parameter

class MatchingAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) 
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]  # batch, vector
        return attn_pool, alpha

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SGConv_Our(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SGConv_Our, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class HeterGConvLayer(torch.nn.Module):
    def __init__(self, feature_size, dropout=0.3, no_cuda=False):
        super(HeterGConvLayer, self).__init__()
        self.no_cuda = no_cuda
        self.hetergconv = SGConv_Our(feature_size, feature_size)

    def forward(self, feature, num_modal, adj_weight):
        if num_modal > 1:
            feature_heter = self.hetergconv(feature, adj_weight)
        else:
            print("Unable to construct heterogeneous graph!")
            feature_heter = feature
        return feature_heter

class HeterGConv_Edge(torch.nn.Module):
    def __init__(self, feature_size, encoder_layer, num_layers, dropout, no_cuda):
        super(HeterGConv_Edge, self).__init__()
        self.num_layers = num_layers
        self.no_cuda = no_cuda

        self.edge_weight = nn.Parameter(torch.ones(500000))

        self.hetergcn_layers = _get_clones(encoder_layer, num_layers)
        self.fc_layer = nn.Sequential(nn.Linear(feature_size, feature_size),
                                      nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc_layers = _get_clones(self.fc_layer, num_layers)

    def forward(self, feature_tuple, dia_lens, win_p, win_f, edge_index=None, commonsense_sim=None):
        num_modal = len(feature_tuple)
        feature = torch.cat(feature_tuple, dim=0)

        if edge_index is None:
            edge_index = self._heter_no_weight_edge(feature, num_modal, dia_lens, win_p, win_f)
        
        edge_weight = self.edge_weight[0:edge_index.size(1)]
        
        # Enhance edge weights with commonsense similarity if provided
        if commonsense_sim is not None:
            # Ensure commonsense_sim matches edge_weight size
            if commonsense_sim.size(0) >= edge_weight.size(0):
                edge_weight = edge_weight * (1 + commonsense_sim[:edge_weight.size(0)])
            else:
                # Pad with ones if not enough similarity values
                padding = torch.ones(edge_weight.size(0) - commonsense_sim.size(0)).type_as(commonsense_sim)
                commonsense_sim_padded = torch.cat([commonsense_sim, padding])
                edge_weight = edge_weight * (1 + commonsense_sim_padded)

        adj_weight = self._edge_index_to_adjacency_matrix(
            edge_index, edge_weight, num_nodes=feature.size(0), no_cuda=self.no_cuda)
        
        feature_sum = feature
        for i in range(self.num_layers):
            feature = self.hetergcn_layers[i](feature, num_modal, adj_weight)
            feature_sum = feature_sum + self.fc_layers[i](feature)
        feat_tuple = torch.chunk(feature_sum, num_modal, dim=0)

        return feat_tuple, edge_index

    def _edge_index_to_adjacency_matrix(self, edge_index, edge_weight=None, num_nodes=100, no_cuda=False):
        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
        else:
            edge_weight = torch.ones(edge_index.size(1)).cuda() if not no_cuda else torch.ones(edge_index.size(1))
        
        adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        adj = adj_sparse.to_dense()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0
        d_inv_sqrt_mat = torch.diag_embed(d_inv_sqrt)
        gcn_fact = torch.matmul(d_inv_sqrt_mat, torch.matmul(adj, d_inv_sqrt_mat))

        if not no_cuda and torch.cuda.is_available():
            gcn_fact = gcn_fact.cuda()

        return gcn_fact

    def _heter_no_weight_edge(self, feature, num_modal, dia_lens, win_p, win_f):
        index_inter = []
        all_dia_len = sum(dia_lens)
        all_nodes = list(range(all_dia_len * num_modal))
        nodes_uni = [None] * num_modal

        for m in range(num_modal):
            nodes_uni[m] = all_nodes[m * all_dia_len:(m + 1) * all_dia_len]

        start = 0
        for dia_len in dia_lens:
            for m, n in permutations(range(num_modal), 2):
                for j, node_m in enumerate(nodes_uni[m][start:start + dia_len]):
                    if win_p == -1 and win_f == -1:
                        nodes_n = nodes_uni[n][start:start + dia_len]
                    elif win_p == -1:
                        nodes_n = nodes_uni[n][start:min(start + dia_len, start + j + win_f + 1)]
                    elif win_f == -1:
                        nodes_n = nodes_uni[n][max(start, start + j - win_p):start + dia_len]
                    else:
                        nodes_n = nodes_uni[n][max(start, start + j - win_p):min(start + dia_len, start + j + win_f + 1)]
                    index_inter.extend(list(product([node_m], nodes_n)))
            start += dia_len
        
        edge_index = (torch.tensor(index_inter).permute(1, 0).cuda() if not self.no_cuda 
                     else torch.tensor(index_inter).permute(1, 0))
        return edge_index

class SenShift_Feat(nn.Module):
    def __init__(self, hidden_dim, dropout, shift_win):
        super().__init__()
        self.shift_win = shift_win
        hidden_dim_shift = 2 * hidden_dim
        self.shift_output_layer = nn.Sequential(nn.Linear(hidden_dim_shift, 2))

    def forward(self, embeds, embeds_temp=None, dia_lens=[]):
        if embeds_temp == None:
            embeds_temp = embeds
        embeds_shift = self._build_match_sample(embeds, embeds_temp, dia_lens, self.shift_win)
        logits = self.shift_output_layer(embeds_shift)
        return logits

    def _build_match_sample(self, embeds, embeds_temp, dia_lens, shift_win):
        start = 0
        embeds_shifts = []
        if shift_win == -1:
            for dia_len in dia_lens:
                embeds_shifts.append(
                    torch.cat([
                        embeds[start:start + dia_len, None, :].repeat(1, dia_len, 1),
                        embeds_temp[None, start:start + dia_len, :].repeat(dia_len, 1, 1),
                    ], dim=-1,).view(-1, 2 * embeds.size(-1)))
                start += dia_len
            embeds_shift = torch.cat(embeds_shifts, dim=0)
        elif shift_win > 0:
            for dia_len in dia_lens:
                win_start = 0
                for i in range(math.ceil(dia_len / shift_win)):
                    if (i == math.ceil(dia_len / shift_win) - 1 and dia_len % shift_win != 0):
                        win = dia_len % shift_win
                    else:
                        win = shift_win
                    embeds_shifts.append(
                        torch.cat([
                            embeds[start + win_start : start + win_start + win, None, :].repeat(1, win, 1),
                            embeds_temp[None, start + win_start : start + win_start + win, :].repeat(win, 1, 1),
                        ], dim=-1,).view(-1, 2 * embeds.size(-1))
                    )
                    win_start += shift_win
                start += dia_len
            embeds_shift = torch.cat(embeds_shifts, dim=0)
        else:
            print("Window must be greater than 0 or equal to -1")
            raise NotImplementedError
        return embeds_shift

class CommonsenseRNNCell(nn.Module):
    def __init__(self, D_m, D_s, D_g, D_p, D_r, D_i, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5, emo_gru=True):
        super(CommonsenseRNNCell, self).__init__()

        self.D_m = D_m
        self.D_s = D_s
        self.D_g = D_g
        self.D_p = D_p
        self.D_r = D_r
        self.D_i = D_i
        self.D_e = D_e

        self.g_cell = nn.GRUCell(D_m+D_p+D_r, D_g)
        self.p_cell = nn.GRUCell(D_s+D_g, D_p)
        self.r_cell = nn.GRUCell(D_m+D_s+D_g, D_r)
        self.i_cell = nn.GRUCell(D_s+D_p, D_i)
        self.e_cell = nn.GRUCell(D_m+D_p+D_r+D_i, D_e)
        
        self.emo_gru = emo_gru
        self.listener_state = listener_state
        if listener_state:
            self.pl_cell = nn.GRUCell(D_s+D_g, D_p)
            self.rl_cell = nn.GRUCell(D_m+D_s+D_g, D_r)

        self.dropout = nn.Dropout(dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, x1, x2, x3, o1, o2, qmask, g_hist, q0, r0, i0, e0):
        """
        U -> batch, D_m
        x1, x2, x3, o1, o2 -> batch, D_m
        x1 -> effect on self; x2 -> reaction of self; x3 -> intent of self
        o1 -> effect on others; o2 -> reaction of others
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)

        ## global state ##
        g_ = self.g_cell(torch.cat([U, q0_sel, r0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        
        ## context ##
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
       
        ## external state ##
        U_r_c_ = torch.cat([U, x2, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1],-1)
        rs_ = self.r_cell(U_r_c_.contiguous().view(-1, self.D_m+self.D_s+self.D_g),
                r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        
        ## internal state ##
        es_c_ = torch.cat([x1, c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(es_c_.contiguous().view(-1, self.D_s+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)

        if self.listener_state:
            ## listener external state ##
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            er_ = o2.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_s)
            ss_ = self._select_parties(rs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_r)
            U_er_ss_ = torch.cat([U_, er_, ss_], 1)
            rl_ = self.rl_cell(U_er_ss_, r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
            
            ## listener internal state ##
            es_ = o1.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_s)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            es_ss_ = torch.cat([es_, ss_], 1)
            ql_ = self.pl_cell(es_ss_, q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
            
        else:
            rl_ = r0
            ql_ = q0
            
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        r_ = rl_*(1-qmask_) + rs_*qmask_            
        
        ## intent ##        
        i_q_ = torch.cat([x3, self._select_parties(q_, qm_idx)], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        is_ = self.i_cell(i_q_.contiguous().view(-1, self.D_s+self.D_p),
                i0.view(-1, self.D_i)).view(U.size()[0], -1, self.D_i)
        il_ = i0
        i_ = il_*(1-qmask_) + is_*qmask_
        
        ## emotion ##        
        es_ = torch.cat([U, self._select_parties(q_, qm_idx), self._select_parties(r_, qm_idx), 
                         self._select_parties(i_, qm_idx)], dim=1) 
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        
        if self.emo_gru:
            e_ = self.e_cell(es_, e0)
        else:
            e_ = es_    
        
        g_ = self.dropout1(g_)
        q_ = self.dropout2(q_)
        r_ = self.dropout3(r_)
        i_ = self.dropout4(i_)
        e_ = self.dropout5(e_)
        
        return g_, q_, r_, i_, e_, alpha

class CommonsenseRNN(nn.Module):
    def __init__(self, D_m, D_s, D_g, D_p, D_r, D_i, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5, emo_gru=True):
        super(CommonsenseRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_r = D_r
        self.D_i = D_i
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = CommonsenseRNNCell(D_m, D_s, D_g, D_p, D_r, D_i, D_e,
                        listener_state, context_attention, D_a, dropout, emo_gru)

    def forward(self, U, x1, x2, x3, o1, o2, qmask):
        """
        U -> seq_len, batch, D_m
        x1, x2, x3, o1, o2 -> seq_len, batch, D_s
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type()) # batch, party, D_p
        r_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_r).type(U.type()) # batch, party, D_r
        i_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_i).type(U.type()) # batch, party, D_i
        
        e_ = torch.zeros(0).type(U.type()) # batch, D_e
        e = e_

        alpha = []
        for u_, x1_, x2_, x3_, o1_, o2_, qmask_ in zip(U, x1, x2, x3, o1, o2, qmask):
            g_, q_, r_, i_, e_, alpha_ = self.dialogue_cell(u_, x1_, x2_, x3_, o1_, o2_, 
                                                            qmask_, g_hist, q_, r_, i_, e_)
            
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e, alpha # seq_len, batch, D_e

class CommonsenseEnhancedGraphSmile(nn.Module):
    def __init__(self, config=None):
        
        super(CommonsenseEnhancedGraphSmile, self).__init__()

        self.embedding_dims = config.embedding_dims
        self.D_s = config.D_s
        self.hidden_dim = config.hidden_dim
        self.n_classes = config.n_classes
        self.heter_n_layers = config.heter_n_layers
        self.win_p = config.win_p
        self.win_f = config.win_f
        self.shift_win = config.shift_win
        self.dropout_rate = config.dropout
        self.dropout_rec_rate = config.dropout_rec
        self.mode1 = config.mode1
        self.norm_strategy = config.norm
        self.listener_state = config.listener_state
        self.context_attention = config.context_attention
        self.emo_gru = config.emo_gru
        self.att2 = config.att2
        self.no_cuda = getattr(config, 'no_cuda', False)
        
        # COSMIC-style dimensions
        D_g = getattr(config, 'D_g', 150)   # Global state
        D_p = getattr(config, 'D_p', 150)   # Party state 
        D_r = getattr(config, 'D_r', 150)   # Reaction state
        D_i = getattr(config, 'D_i', 150)   # Intent state
        D_a_att = getattr(config, 'D_a_att', 100)  # Attention dimension
        
        D_e = D_p + D_r + D_i  # Emotion state (unidirectional)
        D_h = self.hidden_dim
        
        # Text feature processing with different fusion modes
        if self.mode1 == 0:
            D_x = 4 * 1024  # Concatenate all 4 variants
        elif self.mode1 == 1:
            D_x = 2 * 1024  # Concatenate first 2 variants
        else:
            D_x = 1024      # Single variant or average
            
        self.linear_in = nn.Linear(D_x, D_h)
        self.r_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        
        # Normalization strategies following COSMIC
        norm_train = True
        self.norm1a = nn.LayerNorm(1024, elementwise_affine=norm_train)
        self.norm1b = nn.LayerNorm(1024, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(1024, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(1024, elementwise_affine=norm_train)
        
        self.norm3a = nn.BatchNorm1d(1024, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(1024, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(1024, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(1024, affine=norm_train)
        
        # COMET feature processing
        self.comet_proj = nn.Linear(768 * 9, self.D_s)
        
        # COSMIC commonsense reasoning components
        self.cs_rnn_f = CommonsenseRNN(
            D_m=D_h, D_s=self.D_s, D_g=D_g, D_p=D_p, D_r=D_r, D_i=D_i, D_e=D_e,
            listener_state=self.listener_state, context_attention=self.context_attention,
            D_a=D_a_att, dropout=self.dropout_rec_rate, emo_gru=self.emo_gru
        )
        self.cs_rnn_r = CommonsenseRNN(
            D_m=D_h, D_s=self.D_s, D_g=D_g, D_p=D_p, D_r=D_r, D_i=D_i, D_e=D_e,
            listener_state=self.listener_state, context_attention=self.context_attention,
            D_a=D_a_att, dropout=self.dropout_rec_rate, emo_gru=self.emo_gru
        )
        
        # Sense GRU for processing COMET features
        self.sense_gru = nn.GRU(input_size=self.D_s, hidden_size=self.D_s//2, num_layers=1, bidirectional=True)
        
        # Text-commonsense fusion
        self.text_commonsense_fusion = nn.Sequential(
            nn.Linear(D_h + 2*D_e, D_h),  # Text + bidirectional emotions
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Note: Cross-modal fusion is handled by GraphSmile's heterogeneous graph convolution
        # No need for separate attention mechanisms
        
        # GraphSmile-style feature projections
        self.dim_layer_v = nn.Sequential(
            nn.Linear(self.embedding_dims[1], self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(self.embedding_dims[2], self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
        )
        
        # GraphSmile heterogeneous graph convolution layers
        hetergconvLayer_tv = HeterGConvLayer(self.hidden_dim, self.dropout_rate, self.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            self.hidden_dim, hetergconvLayer_tv, self.heter_n_layers[0], self.dropout_rate, self.no_cuda
        )
        hetergconvLayer_ta = HeterGConvLayer(self.hidden_dim, self.dropout_rate, self.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            self.hidden_dim, hetergconvLayer_ta, self.heter_n_layers[1], self.dropout_rate, self.no_cuda
        )
        hetergconvLayer_va = HeterGConvLayer(self.hidden_dim, self.dropout_rate, self.no_cuda)
        self.hetergconv_va = HeterGConv_Edge(
            self.hidden_dim, hetergconvLayer_va, self.heter_n_layers[2], self.dropout_rate, self.no_cuda
        )
        
        # GraphSmile modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        
        # COSMIC-style final classification
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, self.n_classes)
        
        # GraphSmile-style additional outputs with COSMIC integration
        # Bottleneck layer for concatenated features (feat_fusion + emotions_flat)
        # feat_fusion: hidden_dim (384), emotions_flat: 2*D_e (900) = total 1284
        bottleneck_input_dim = self.hidden_dim + 2*D_e  
        self.emo_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.sen_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.emo_output = nn.Linear(self.hidden_dim, self.n_classes)
        self.sen_output = nn.Linear(self.hidden_dim, 3)
        
        # Sentiment shift detection
        self.senshift = SenShift_Feat(self.hidden_dim, self.dropout_rate, self.shift_win)
        
        # Learnable weighted fusion for 6 modal features
        self.fusion_weights = nn.Parameter(torch.ones(6))
        
        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_rec = nn.Dropout(self.dropout_rec_rate)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def _batch_to_all(self, features, dia_lengths):
        """
        Extract valid utterances from padded sequences (GraphSmile's batch_to_all_tva approach)
        features -> seq_len, batch, feature_dim
        dia_lengths -> list of actual dialogue lengths
        """
        node_features = []
        batch_size = features.size(1)
        
        for j in range(batch_size):
            # Extract only valid utterances for this dialogue
            node_features.append(features[:dia_lengths[j], j, :])
        
        # Concatenate all valid utterances
        return torch.cat(node_features, dim=0)

    def _compute_commonsense_similarity(self, commonsense_states, dia_lengths):
        """
        Compute commonsense similarity for edge enhancement
        """
        # Extract valid commonsense states using dia_lengths
        valid_states = self._batch_to_all(commonsense_states, dia_lengths)
        
        # Compute pairwise cosine similarity
        normalized_states = F.normalize(valid_states, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_states, normalized_states.t())
        
        # Return flattened similarities (will be truncated to match edge count in HeterGConv_Edge)
        return similarity_matrix.flatten()

    def forward(self, roberta_features, comet_features, visual_features, audio_features, 
                speakers, qmask, umask, dia_lengths, att2=None, return_hidden=False):
        """
        Commonsense-enhanced forward pass
        """
        if att2 is None:
            att2 = self.att2
            
        seq_len, batch_size, feature_dim = roberta_features[0].shape
        
        # 1. COSMIC-style RoBERTa feature processing
        r1, r2, r3, r4 = roberta_features
        
        if self.norm_strategy == 1:
            r1 = self.norm1a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm1b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm1c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm1d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 2:
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
        elif self.norm_strategy == 3:
            r1 = self.norm3a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm3b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm3c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm3d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        
        # RoBERTa fusion based on mode1
        if self.mode1 == 0:
            r = torch.cat([r1, r2, r3, r4], axis=-1)
        elif self.mode1 == 1:
            r = torch.cat([r1, r2], axis=-1)
        elif self.mode1 == 2:
            r = (r1 + r2 + r3 + r4)/4
        elif self.mode1 == 3:
            r = r1
        elif self.mode1 == 4:
            r = r2
        elif self.mode1 == 5:
            r = r3
        elif self.mode1 == 6:
            r = r4
        elif self.mode1 == 7:
            r = self.r_weights[0]*r1 + self.r_weights[1]*r2 + self.r_weights[2]*r3 + self.r_weights[3]*r4
            
        text_projected = self.linear_in(r)  # [seq_len, batch, hidden_dim]
        
        # 2. Process COMET features
        comet_concat = torch.cat(comet_features, dim=-1)        # [seq_len, batch, 768*9]
        comet_proj = self.comet_proj(comet_concat)              # [seq_len, batch, D_s]
        
        # Extract COMET reasoning components
        x1 = comet_features[4]  # xEffect (effect on self)
        x2 = comet_features[5]  # xReact (reaction of self)  
        x3 = comet_features[0]  # xIntent (intent of self)
        o1 = comet_features[6]  # oWant (want of others)
        o2 = comet_features[8]  # oReact (reaction of others)
        
        # 3. COSMIC commonsense reasoning
        emotions_f, alpha_f = self.cs_rnn_f(text_projected, x1, x2, x3, o1, o2, qmask)
        
        # Process COMET features through sense GRU
        out_sense, _ = self.sense_gru(x1)
        
        # Reverse sequences for backward pass
        rev_text = self._reverse_seq(text_projected, umask)
        rev_x1 = self._reverse_seq(x1, umask)
        rev_x2 = self._reverse_seq(x2, umask)
        rev_x3 = self._reverse_seq(x3, umask)
        rev_o1 = self._reverse_seq(o1, umask)
        rev_o2 = self._reverse_seq(o2, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        
        emotions_b, alpha_b = self.cs_rnn_r(rev_text, rev_x1, rev_x2, rev_x3, rev_o1, rev_o2, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        
        # Ensure both emotion tensors have the same dimensions before concatenation
        # Check all dimensions: [seq_len, batch, feature_dim]
        if emotions_f.shape != emotions_b.shape:
            # Get the maximum size for each dimension
            max_seq_len = max(emotions_f.size(0), emotions_b.size(0))
            max_batch_size = max(emotions_f.size(1), emotions_b.size(1))
            feature_dim = emotions_f.size(2)  # This should be the same
            
            # Pad emotions_f if needed
            if emotions_f.size(0) < max_seq_len or emotions_f.size(1) < max_batch_size:
                pad_f = torch.zeros(max_seq_len, max_batch_size, feature_dim).type_as(emotions_f)
                pad_f[:emotions_f.size(0), :emotions_f.size(1), :] = emotions_f
                emotions_f = pad_f
            
            # Pad emotions_b if needed  
            if emotions_b.size(0) < max_seq_len or emotions_b.size(1) < max_batch_size:
                pad_b = torch.zeros(max_seq_len, max_batch_size, feature_dim).type_as(emotions_b)
                pad_b[:emotions_b.size(0), :emotions_b.size(1), :] = emotions_b
                emotions_b = pad_b
        
        # Concatenate bidirectional emotions
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)  # [seq_len, batch, 2*D_e]
        emotions = self.dropout_rec(emotions)
        
        # 4. Enhance text with commonsense knowledge
        enhanced_text = self.text_commonsense_fusion(
            torch.cat([text_projected, emotions], dim=-1)
        )
        
        # 5. Use COSMIC's enhanced text (text + commonsense) as final text representation
        # No need for additional cross-modal attention here - GraphSmile handles cross-modal fusion via heterogeneous graphs
        final_text = enhanced_text
        
        # 6. Process visual and audio features (GraphSmile style)
        visual_proj = self.dim_layer_v(visual_features)         # [seq_len, batch, hidden_dim]
        audio_proj = self.dim_layer_a(audio_features)           # [seq_len, batch, hidden_dim]
        
        # 7. GraphSmile heterogeneous graph processing
        # Extract only valid utterances from padded sequences (following GraphSmile's batch_to_all_tva)
        final_text_flat = self._batch_to_all(final_text, dia_lengths)
        visual_proj_flat = self._batch_to_all(visual_proj, dia_lengths) 
        audio_proj_flat = self._batch_to_all(audio_proj, dia_lengths)
        
        # Compute commonsense similarity for edge enhancement
        commonsense_sim = self._compute_commonsense_similarity(emotions, dia_lengths)
        
        # Enhanced heterogeneous graph convolution with commonsense
        featheter_tv, heter_edge_index = self.hetergconv_tv(
            (final_text_flat, visual_proj_flat), dia_lengths, self.win_p, self.win_f, 
            commonsense_sim=commonsense_sim
        )
        featheter_ta, heter_edge_index = self.hetergconv_ta(
            (final_text_flat, audio_proj_flat), dia_lengths, self.win_p, self.win_f, heter_edge_index
        )
        featheter_va, heter_edge_index = self.hetergconv_va(
            (visual_proj_flat, audio_proj_flat), dia_lengths, self.win_p, self.win_f, heter_edge_index
        )
        
        # 8. Learnable weighted modal fusion (improved from GraphSmile's averaging)
        features = [
            self.modal_fusion(featheter_tv[0]),  # Enhanced Text from Text-Visual
            self.modal_fusion(featheter_ta[0]),  # Enhanced Text from Text-Audio  
            self.modal_fusion(featheter_tv[1]),  # Enhanced Visual from Text-Visual
            self.modal_fusion(featheter_va[0]),  # Enhanced Visual from Visual-Audio
            self.modal_fusion(featheter_ta[1]),  # Enhanced Audio from Text-Audio
            self.modal_fusion(featheter_va[1])   # Enhanced Audio from Visual-Audio
        ]
        stacked = torch.stack(features)  # shape: [6, N, D]
        weighted = torch.einsum('i,ijd->jd', self.fusion_weights, stacked)
        feat_fusion = weighted / self.fusion_weights.sum()
        
        # 9. Final classification
        # COSMIC-style classification using bidirectional emotions
        alpha = []
        if att2:
            att_emotions = []
            alpha = []
            # Create proper mask for emotions (which may have different batch size due to flattening)
            seq_len, batch_size = emotions.shape[:2]
            emotion_mask = torch.ones(batch_size, seq_len).to(emotions.device)
            
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=emotion_mask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
            
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        
        # 10. Integrate COSMIC commonsense with GraphSmile multimodal fusion
        # Flatten COSMIC emotions to match feat_fusion (utterance-level predictions)
        emotions_flat = self._batch_to_all(emotions, dia_lengths)  # Same flattening as feat_fusion
        
        # Concatenate: [GraphSmile multimodal fusion] + [COSMIC commonsense emotions]
        final_features = torch.cat([feat_fusion, emotions_flat], dim=-1)  # [N_utterances, 2*hidden_dim]
        
        # Enhanced classification with both multimodal and commonsense information
        logit_emo = self.emo_output(self.emo_bottleneck(final_features))  # Emotion: multimodal + commonsense
        logit_sen = self.sen_output(self.sen_bottleneck(final_features))   # Sentiment: multimodal + commonsense  
        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths) # Shift: uses multimodal only
        
        
        if return_hidden:
            return hidden, alpha, alpha_f, alpha_b, emotions
        
        return log_prob, out_sense, alpha, alpha_f, alpha_b, emotions, logit_emo, logit_sen, logit_shift, feat_fusion