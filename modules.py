# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

import pdb

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads))
        d_q = hp.token_embedding_size // 2
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=hp.token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out

class TransformerStyleTokenLayer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)

        self.context_gru = nn.GRU(input_size=hp.encoder_embedding_dim, 
                hidden_size=hp.encoder_embedding_dim, batch_first=True)
#        self.reference_gru = nn.GRU(input_size=hp.encoder_embedding_dim,
#                hidden_size=hp.encoder_embedding_dim, batch_first=True)
        
        self.tfs_type = hp.tfs_type
        if self.tfs_type == 'dual':
            self.mab1 = MAB_qkv(hp.encoder_embedding_dim,
                    hp.encoder_embedding_dim,
                    hp.ref_enc_gru_size,
                    hp.token_embedding_size//2, p=0.5, num_heads=hp.num_heads)
            self.mab2 = MAB_qkv(hp.encoder_embedding_dim,
                    hp.encoder_embedding_dim,
                    hp.ref_enc_gru_size,
                    hp.token_embedding_size//2, p=0.5, num_heads=hp.num_heads)
        elif self.tfs_type == 'single':
            self.mab1 = MAB_qkv(hp.encoder_embedding_dim,
                    hp.encoder_embedding_dim,
                    hp.ref_enc_gru_size,
                    hp.token_embedding_size, p=0.5, num_heads=hp.num_heads)


    def forward(self, text, text_len, rmel, rtext, rtext_len):
        mel_emb = self.encoder(rmel)
        mel_emb = mel_emb.unsqueeze(1)
        mel_emb = mel_emb.transpose(0,1).repeat(text.size(0),1,1) # bsz, bsz_s, d
        
        text_len = text_len.cpu().numpy()
        _tp = nn.utils.rnn.pack_padded_sequence(
                text, text_len, batch_first=True)
        self.context_gru.flatten_parameters()
        _, query = self.context_gru(_tp)

        rtext_len = rtext_len.cpu().numpy()
        _tp = nn.utils.rnn.pack_padded_sequence(
                rtext, rtext_len, batch_first=True)
        self.context_gru.flatten_parameters()
        _, key = self.context_gru(_tp)

        if self.tfs_type == 'single':
            st, attn = self.mab1(query.transpose(0,1),
                    key.repeat(text.size(0),1,1),
                    mel_emb, get_attn=True)

#            Q = query.transpose(0,1) # bsz,1,d
#            K = key.repeat(text.size(0), 1, 1).transpose(1,2) # bsz,d,bsz_s
#            mattn = (Q @ K) / math.sqrt(Q.size(-1))
#            _attn = attn.reshape(st.size(0),-1,attn.size(-1)).mean(1)
#            pdb.set_trace()

            return st.repeat(1, text.size(1), 1) 


        
#        if self.tfs_type == 'dual':
#            _, query = self.context_gru(context)  # 1,bsz,d
#            _, key = self.context_gru(refcontext) # 1,bszs,d
#            # change to cat later
#
#            st1, attn = self.mab1(query.transpose(0,1),
#                    key.repeat(context.size(0),1,1),
#                    mel_emb, get_attn=True) # (bsz,1,d), global style
#
#            st1 = st1.repeat(1, context.size(1), 1)
#            
#            st2 = self.mab2(context,
#                    key.repeat(context.size(0),1,1),
#                    mel_emb) # token-wise style
#            return torch.cat((st1, st2), dim=-1)
#
#        elif self.tfs_type == 'single':
#            _, query = self.context_gru(context)  # 1,bsz,d
#            _, key = self.context_gru(refcontext) # 1,bszs,d
#            # change to cat later
#
#            st, attn = self.mab1(query.transpose(0,1),
#                    key.repeat(context.size(0),1,1),
#                    mel_emb, get_attn=True) # (bsz,1,d)
#            _attn = attn.reshape(st.size(0),-1,attn.size(-1)).mean(1)
#            pdb.set_trace()
#            return st.repeat(1,context.size(1),1)
#

        
class TST(nn.Module):
    # TransformerStyleToken
    def __init__(self, hp):
        super().__init__()
        self.encoder_mel = ReferenceEncoder(hp) # output: ref_enc_gru_size
#        self.isab = ISAB(hp.token_embedding_size // 2, hp.token_embedding_size, hp.token_num,
#                num_heads=hp.num_heads, p=0.5)
        self.mab = MAB(hp.encoder_embedding_dim, hp.ref_enc_gru_size, hp.token_embedding_size, p=0.5)
        
        if hp.context_gru:
            self.context_gru = nn.GRU(input_size=hp.encoder_embedding_dim,
                    hidden_size=hp.encoder_embedding_dim, batch_first=True)
        else:
            self.context_gru = None

    def forward(self, refmel, context):
        '''
        refmel: bsz_support, 80, T_signal
        context: bsz, T_text, encoder_embedding_dim
        '''

        # context (text_embedding): (bsz, t, d)
        mel_emb = self.encoder_mel(refmel) # (bsz_s, d)
        #mel_emb = self.isab(mel_emb.unsqueeze(1)) # (bsz_s, 1, d)
        mel_emb = mel_emb.unsqueeze(1)
        mel_emb = mel_emb.transpose(0,1).repeat(context.size(0),1,1) # bsz, bsz_s, d

        if self.context_gru is None:
            st, attn = self.mab(context, mel_emb, get_attn=True) # (bsz,1,d)
#            _attn = attn.reshape(st.size(0),-1,attn.size(-1)).mean(1)
#            pdb.set_trace()
            return st
        else:
            _, ctxh = self.context_gru(context)
            st, attn = self.mab(ctxh.transpose(0,1), mel_emb, get_attn=True) # (bsz,1,d)
#            _attn = attn.reshape(st.size(0),-1,attn.size(-1)).mean(1)
#            pdb.set_trace()
            return st.repeat(1,context.size(1),1)

class GST(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)

    def forward(self, inputs):
        # inputs.shape: (bsz, 80, T)
        enc_out = self.encoder(inputs)
        # enc_out.shape (bsz, 128)
        style_embed = self.stl(enc_out)
        # style_ebmed.shape: (bsz, 1, 256)

        return style_embed

class MAB(nn.Module):
    def __init__(self, dim_X, dim_Y, dim, num_heads=4, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_X, dim)
        self.fc_k = nn.Linear(dim_Y, dim)
        self.fc_v = nn.Linear(dim_Y, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()

    def forward(self, X, Y, mask=None, get_attn=False):
        Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)

        A_logits = (Q_ @ K_.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        if mask is not None:
            mask = torch.stack([mask]*Q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, 0)
            A_logits.masked_fill_(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
            # to prevent underflow due to no attention
            A.masked_fill_(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        
        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.ln1(Q + self.dropout1(attn))
        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        if get_attn:
            return O, A
        return O

class MAB_qkv(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim, num_heads=4, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim)
        self.fc_k = nn.Linear(dim_k, dim)
        self.fc_v = nn.Linear(dim_v, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()


    def forward(self, query, key, value, mask=None, get_attn=False):
        Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)

        A_logits = (Q_ @ K_.transpose(-2, -1)) /  math.sqrt(Q.shape[-1])
        if mask is not None:
            mask = torch.stack([mask]*Q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, 0)
            A_logits.masked_fill_(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
            # to prevent underflow due to no attention
            A.masked_fill_(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        
        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.ln1(Q + self.dropout1(attn))
        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        if get_attn:
            return O, A
        return O


class SAB(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.mab = MAB(dim_X, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class StackedSAB(nn.Module):
    def __init__(self, dim_X, dim, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [SAB(dim_X, dim, **kwargs)] + \
                [SAB(dim, dim, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for sab in self.blocks:
            X = sab(X, mask=mask)
        return X

class PMA(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        I = self.I if X.dim() == 2 else self.I.repeat(X.shape[0], 1, 1)
        return self.mab(I, X, mask=mask)

class ISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.pma = PMA(dim_X, dim, num_inds, **kwargs)
        self.mab = MAB(dim_X, dim, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, self.pma(X, mask=mask))

class StackedISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [ISAB(dim_X, dim, num_inds, **kwargs)] + \
                [ISAB(dim, dim, num_inds, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for isab in self.blocks:
            X = isab(X, mask=mask)
        return X

class aPMA(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.I0 = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.I0)
        self.pma = PMA(dim, dim, 1, **kwargs)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, num_iters):
        I = self.I0
        for i in range(1, num_iters):
            I = torch.cat([I, self.pma(I)], 1)
        return self.mab(I.repeat(X.shape[0], 1, 1), X)
