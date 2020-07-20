# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, TransformerModel, base_architecture, transformer_vaswani_wmt_en_fr_big
import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_tsa')
class TransformerTSAModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)


    def compute(self, src_tokens, src_lengths, prev_output_tokens, return_content_mask=None, query=None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = encode_two_stream_attention(self.decoder, 
                prev_output_tokens, 
                encoder_out=encoder_out, 
                return_content_mask=return_content_mask, 
                query=query, **kwargs)
        return decoder_out


def encode_two_stream_attention(self,
    prev_output_tokens, 
    encoder_out=None,
    return_content_mask=None,
    query=None,
):
    sz = prev_output_tokens.size()
    positions = self.embed_positions(prev_output_tokens)

    q = prev_output_tokens.new_full(prev_output_tokens.size(), self.mask_idx)

    c = self.embed_scale * self.embed_tokens(prev_output_tokens) + positions
    q = self.embed_scale * self.embed_tokens(q) + positions
    x = torch.cat((c, q), dim=1)

    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x.transpose(0, 1)
    c, q = x[:sz[1]], x[sz[1]:]

    self_attn_padding_mask = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    for layer in self.layers:
        encoder_state = encoder_out.encoder_out
        self_attn_mask = self.buffered_future_mask(c)

        c, q, layer_attn = attention(
            layer,
            c,
            q,
            encoder_state,
            encoder_out.encoder_padding_mask if encoder_out is not None else None,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )

    if self.layer_norm:
        q = self.layer_norm(q)
    q = q[1:].transpose(0, 1).contiguous()
    q = self.output_layer(q)
    
    if return_content_mask is not None:
        if self.layer_norm:
            c = self.layer_norm(c)
        c = c[1:].transpose(0, 1).contiguous()
        c = self.output_layer(c)
        #c = self.output_layer(c[return_content_mask, :])

    return q, c


def attention(
    self,
    c,
    q,
    encoder_out=None,
    encoder_padding_mask=None,
    self_attn_mask=None,
    self_attn_padding_mask=None,
):
    def reuse_fn(x, residual):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
 
        if self.encoder_attn:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
             
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=False,
            )
    
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
    
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    residual_q = q
    residual_c = c

    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)
    
    q, c, _ = two_stream_attention(
        self.self_attn,
        query=q,
        content=c,
        key=c,
        value=c,
        key_padding_mask=self_attn_padding_mask,
        attn_mask=self_attn_mask,
    )

    c, attn = reuse_fn(c, residual_c)
    q, attn = reuse_fn(q, residual_q)
    return c, q, attn


def two_stream_attention(
    self,
    query, content, key, value,
    key_padding_mask=None, 
    attn_mask=None,
):
    src_len, bsz, embed_dim = query.size(0), query.size(1), query.size(2)

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)

    def transpose_fn(x):
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def attn_fn(attn_weights, is_query=False):
        if attn_mask is not None:
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, src_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, src_len, src_len)
        if is_query is True:
            query_mask = torch.eye(attn_weights.size(-1)).to(attn_weights) * -1e9
            query_mask[0][0] = 0.0
            attn_weights = attn_weights + query_mask
        attn_weights = utils.softmax(
            attn_weights, dim=-1,        
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        return attn_weights

    def out_fn(attn_weights, v, q_v=None):
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        return self.out_proj(attn)

    q = transpose_fn(self.scaling * self.q_proj(query))
    c = transpose_fn(self.scaling * self.q_proj(content))
    k = transpose_fn(self.k_proj(key))
    v = transpose_fn(self.v_proj(value))
    
    c_attn_weight = torch.bmm(c, k.transpose(1, 2))
    q_attn_weight = torch.tril(torch.bmm(q, k.transpose(1, 2)), -1)

    c_attn_weight = attn_fn(c_attn_weight)
    q_attn_weight = attn_fn(q_attn_weight, is_query=True)

    c = out_fn(c_attn_weight, v)
    q = out_fn(q_attn_weight, v)

    return q, c, q_attn_weight



@register_model_architecture('transformer_tsa', 'tsa_small')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)

@register_model_architecture('transformer_tsa', 'tsa_large_enfr')
def tsa_large_enfr(args):
    transformer_vaswani_wmt_en_fr_big(args)
