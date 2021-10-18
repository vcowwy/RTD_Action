import copy
from typing import Optional

import paddle
from paddle import nn
import paddle.nn.functional as F
from x2paddle import torch2paddle
from x2paddle.torch2paddle import create_tensor


class Transformer(nn.Layer):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        num_encoder_layers = 3
        print('Use {}-layer MLP as encoder'.format(num_encoder_layers))
        self.encoder = simpleMLP(d_model * 2, d_model,
                                 d_model, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                          decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                paddle.nn.initializer.XavierNormal(p)

    def forward(self, src, query_embed, pos_embed):
        bs, t, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = paddle.full_like(query_embed, fill_value=0.0).requires_grad_(False)

        src = paddle.concat([src, pos_embed], axis=2)
        memory = self.encoder(src)
        hs = self.decoder(tgt.permute(1, 0, 2),
                          memory.permute(1, 0, 2),
                          memory_key_padding_mask=None,
                          pos=pos_embed.permute(1, 0, 2),
                          query_pos=query_embed.permute(1, 0, 2))
        return hs, memory.permute(1, 2, 0).view(bs, c, t)


class simpleMLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Layer):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[create_tensor]=None,
                memory_mask: Optional[create_tensor]=None,
                tgt_key_padding_mask: Optional[create_tensor]=None,
                memory_key_padding_mask: Optional[create_tensor]=None,
                pos: Optional[create_tensor]=None,
                query_pos: Optional[create_tensor]=None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = paddle.nn.MultiHeadAttention(d_model,
                                                           nhead,
                                                           dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[create_tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[create_tensor]=None,
                     memory_mask: Optional[create_tensor]=None,
                     tgt_key_padding_mask: Optional[create_tensor]=None,
                     memory_key_padding_mask: Optional[create_tensor]=None,
                     pos: Optional[create_tensor]=None,
                     query_pos: Optional[create_tensor]=None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[create_tensor]=None,
                    memory_mask: Optional[create_tensor]=None,
                    tgt_key_padding_mask: Optional[create_tensor]=None,
                    memory_key_padding_mask: Optional[create_tensor]=None,
                    pos: Optional[create_tensor]=None,
                    query_pos: Optional[create_tensor]=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[create_tensor] = None,
                memory_mask: Optional[create_tensor] = None,
                tgt_key_padding_mask: Optional[create_tensor] = None,
                memory_key_padding_mask: Optional[create_tensor] = None,
                pos: Optional[create_tensor] = None,
                query_pos: Optional[create_tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(d_model=args.hidden_dim,
                       dropout=args.dropout,
                       nhead=args.nheads,
                       dim_feedforward=args.dim_feedforward,
                       num_encoder_layers=args.enc_layers,
                       num_decoder_layers=args.dec_layers,
                       normalize_before=args.pre_norm,
                       return_intermediate_dec=True)


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return paddle.nn.functional.gelu
    if activation == 'glu':
        return paddle.nn.functional.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')
