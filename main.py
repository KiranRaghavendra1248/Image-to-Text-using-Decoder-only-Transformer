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

from utils import *
from models import *
from dataset import *

# Tokenizer using spacy
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

train_df = create_dataframe("/kaggle/input/coco-2017-dataset/coco2017","captions_train2017.json","train2017")
val_df = create_dataframe("/kaggle/input/coco-2017-dataset/coco2017","captions_val2017.json","val2017")

train_df = train_df.sample(40000)
train_df = train_df.reset_index(drop=True)

val_df  =val_df.sample(10000)
val_df = val_df.reset_index(drop=True)

# Clean text in train and val dataframe
train_df['caption'] = train_df['caption'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x))])
val_df['caption'] = val_df['caption'].apply(lambda x: [token.text.lower() for token in tokenizer(text_cleaner(x))])

# Add START AND END tokens to summary
train_df['caption'] = train_df['caption'].apply(lambda x : ['_START_']+ x + ['_END_'])
val_df['caption'] = val_df['caption'].apply(lambda x : ['_START_']+ x + ['_END_'])

# Build vocabularies - each word has an index, note : words sorted in ascending order
all_tokens = train_df['caption'].tolist() + val_df['caption'].tolist()
target_vocab = {actual_word: idx for idx, (word_num, actual_word) in enumerate(sorted(enumerate(set(token for tokens in all_tokens for token in tokens)), key=lambda x: x[1]))}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using",device)

transform =transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# Create custom datasets
train_dataset = CustomDataset(train_df, target_vocab, transform)
val_dataset = CustomDataset(val_df, target_vocab, transform)

tgt_vocab_size = len(target_vocab)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = get_max_seqlen()
dropout = 0.1
num_workers = 2
num_epochs = 5

resnet_encoder = get_resnet_encoder(d_model,pretrained=True)
model = Encoder_Decoder(resnet_encoder, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
print(model)

# Specify optimizer and loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

train_loop(model,train_loader,criterion,optimizer,device)
test_loop(model,val_loader,criterion,device)