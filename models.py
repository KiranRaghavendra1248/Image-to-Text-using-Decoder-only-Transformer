# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
import random
from tqdm import tqdm
from PIL import Image
import math
import json
from collections import defaultdict


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dim_perhead = embedding_dim // num_heads

        self.W_q = nn.Linear(embedding_dim,embedding_dim)
        self.W_k = nn.Linear(embedding_dim,embedding_dim)
        self.W_v = nn.Linear(embedding_dim,embedding_dim)
        self.W_o = nn.Linear(embedding_dim,embedding_dim)

    def scaled_dot_product_attention(self, Q, K,V,mask=None):
        # Q,K,V Shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]

        K = K.transpose(-2,-1)  # K = K.permute(0,1,3,2) also works
        # K Shape(after permute) : [Batch_Size X Num_Heads X Dim Per Head X Seq_len]
        attn_scores = torch.matmul(Q,K) / math.sqrt(self.dim_perhead)
        # attn_scores Shape : [Batch_Size X Num_Heads X Seq_len X Seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores,dim=-1)
        # attn_probs Shape : [Batch_Size X Num_Heads X Seq_len X Seq_len]
        output = torch.matmul(attn_probs, V)
        # output Shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]
        return output

    def split_heads(self, x):
        # X shape : [Batch_Size X Seq_len X Embedding Dim]
        batch_size, seq_length, d_model = x.size()
        x = x.view(batch_size, seq_length,self.num_heads,self.dim_perhead)
        # X shape : [Batch_Size X Seq_len X Num_Heads X Dim Per Head]
        x = x.transpose(1, 2)
        # X shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]
        return x

    def combine_heads(self, x):
        # x Shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]
        batch_size, _, seq_length, dim_perhead = x.size()
        x = x.transpose(1, 2).contiguous()
        # x Shape : [Batch_Size X Seq_len X Num_Heads X Dim Per Head]
        x = x.view(batch_size, seq_length,self.embedding_dim)
        # x Shape : [Batch_Size X Seq_len X Embedding Dim]
        return x

    def forward(self, Q, K, V, mask=None):
        # Q,K,V Shape : [Batch_Size X Seq_len X Embedding Dim]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # Q,K,V Shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output Shape : [Batch_Size X Num_Heads X Seq_len X Dim Per Head]
        output = self.W_o(self.combine_heads(attn_output))
        # output Shape :  # x Shape : [Batch_Size X Seq_len X Embedding Dim]
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # shape does not change here
        return self.fc2(F.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self, x):
        # shape does not change here, adding positional encoding information
        return x + self.pe[:, :x.size(1)]

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # shape does not change here
        return self.fc2(F.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self, x):
        # shape does not change here, adding positional encoding information
        return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x,enc_output,enc_output,src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Encoder_Decoder(nn.Module):
    def __init__(self, resnet_encoder,tgt_vocab_size, d_model,num_heads, num_layers, d_ff,max_seq_length, dropout):
        super(Encoder_Decoder, self).__init__()
        self.encoder = resnet_encoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(
            d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])

        self.fc = nn.Linear(d_model,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    def generate_mask(self, src, tgt):
        src_mask = None
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length,),diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, img, caption):
        # img shape : [batch_size X 3 X 512 X 512]  ,  caption shape : [batch_size X seq_len]
        src_mask, caption_mask = self.generate_mask(img, caption)
        caption_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(caption)))
        # caption_embedded : [batch_size X seq_len X embedding_dim]

        enc_output = self.encoder(img)
        # enc_output shape : [batch_size X 512]
        enc_output = enc_output.unsqueeze(1)
        # enc_output shape : [batch_size X 1 X 512]
        dec_output = caption_embedded
        for dec_layer in self.decoder_layers:dec_output = dec_layer(dec_output,enc_output,src_mask,caption_mask)
        output = self.fc(dec_output)
        return output
