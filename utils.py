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

# Check accuracy function
def check_accuracy(output,labels):
    _ , predpos = output.max(1)
    num_samples=len(labels)
    num_correct=(predpos==labels).sum()
    return (num_correct/num_samples)*100

# Save checkpoint
def save_checkpoint(state,filename='weights.pth.tar'):
    print('Saving weights-->')
    torch.save(state,filename)

# Load checkpoint
def load_checkpoint(checkpoint,model,optim):
    print('Loading weights-->')
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])

def create_dataframe(BASE_PATH,json_file,image_folder):
    path = os.path.join(BASE_PATH,"annotations/"+json_file)
    with open(path) as f:
        data = json.load(f)
        data = data['annotations']

    img_cap_pairs = []

    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    captions['image'] = captions['image'].apply(
        lambda x: f'{BASE_PATH}/{image_folder}/{x}'
    )
    captions = captions.reset_index(drop=True)
    return captions

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


def text_cleaner(text):
    newString = text.lower()
    newString = newString.replace('"', "'")
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    tokens = [w for w in newString.split()]
    return " ".join(tokens)

def get_max_seqlen(train_df, val_df):
    max_length = 0
    for index, row in train_df.iterrows():
        # Calculate the length of the current row
        row_length = len(row['caption'])
        # Update the maximum length if the current row length is greater
        max_length = max(max_length, row_length)
    for index, row in val_df.iterrows():
        # Calculate the length of the current row
        row_length = len(row['caption'])
        # Update the maximum length if the current row length is greater
        max_length = max(max_length, row_length)
    print("Max length in dataset ",max_length)
    return max_length

# Define collate function for DataLoader
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images,dim=0)
    padded_captions = pad_sequence(captions, batch_first=True)
    return images, padded_captions

# Use pretrained resnet101 as feature extractor
def get_resnet_encoder(out_features,pretrained=True):
    resnet_encoder = torchvision.models.resnet101(pretrained=pretrained)
    # Modify this model to encode feature to embedding_dim = 512, so that the
    # image feature encoding can be used for Cross Attention in Decoder only Transformer
    resnet_encoder.fc = nn.Linear(in_features = 2048, out_features = out_features)
    return resnet_encoder


def train_loop(model, dataloader, loss_fun,optimizer, device, num_epochs, scheduler):
    model.train()
    model.to(device)
    min_loss = None
    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        loop = tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
        for batch, (x, y) in loop:
            # put on cuda
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x, y)

            # calculate loss & accuracy
            loss = loss_fun(y_pred.reshape(-1,len(target_vocab)),y.reshape(-1))
            losses.append(loss.detach().item())

            accuracy = check_accuracy(y_pred.reshape(-1,len(target_vocab)),y.reshape(-1))
            accuracies.append(accuracy.detach().item())

            # zero out prior gradients
            optimizer.zero_grad()

            # backprop
            loss.backward()

            # update weights
            optimizer.step()
            scheduler.step()

            # Update TQDM progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}] ")
            loop.set_postfix(loss=loss.detach().item(),accuracy=accuracy.detach().item())

        moving_loss = sum(losses) / len(losses)
        moving_accuracy = sum(accuracies) / len(accuracies)
        checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
        # Save check point
        if min_loss == None:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        elif moving_loss < min_loss:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        print('Epoch {0} : Loss = {1} , Training Accuracy={2}'.format(epoch, moving_loss,moving_accuracy))


def test_loop(model, dataloader, loss_fun,device):
    model.eval()
    model.to(device)
    losses = []
    samples, correct = 0, 0
    loop = tqdm(enumerate(dataloader),total=len(dataloader), leave=True)
    with torch.no_grad():
        for batch, (x, y) in loop:
            # put on cuda
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x, y)

            # caclulate test loss
            loss = loss_fun(y_pred.reshape(-1,len(target_vocab)),y.reshape(-1))
            losses.append(loss.detach().item())

            # accuracy over entire dataset
            _, predpos = y_pred.reshape(-1,len(target_vocab)).max(1)
            samples += len(y.reshape(-1))
            correct += (predpos == y.reshape(-1)).sum().item()

            # Update TQDM progress bar
            loop.set_postfix(loss=loss.item())

    print("Final Test Accuracy = ",100 * (correct / samples))

