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

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_vocab, image_transform=None):
        self.dataframe = dataframe
        self.target_vocab = target_vocab
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path=self.dataframe.loc[idx]['image']
        image=Image.open(img_path)
        caption = [self.target_vocab[word] for word in self.dataframe.loc[idx]['caption']]
        if self.image_transform:
            image = self.image_transform(image)
        if image.shape[0] != 3:
            return torch.randn((3,512,512)),torch.tensor(caption)
        return image,torch.tensor(caption)