import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
from torch.nn import LSTM
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

"""## Encoder"""

class Image_Encoder(nn.Module):

    def __init__(self, embed_dim):

        super(Image_Encoder, self).__init__()
        self.model = models.vgg19(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1]) # remove vgg19 last layer
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, image):

        with torch.no_grad():
            img_feature = self.model(image) # (batch, channel, height, width)
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1).detach()
        return l2_norm

class Question_Encoder(nn.Module):

    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):

        super(Question_Encoder, self).__init__()
        self.word_embedding = nn.Embedding(qu_vocab_size, word_embed)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden) # input_feature, hidden_feature, num_layer
        self.fc = nn.Linear(2*num_hidden*hidden_size, qu_feature_size)

    def forward(self, question):

        qu_embedding = self.word_embedding(question)                # (batchsize, qu_length=30, word_embed=300)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)                 # (qu_length=30, batchsize, word_embed=300)
        _, (hidden, cell) = self.lstm(qu_embedding)                 # (num_layer=2, batchsize, hidden_size=1024)
        qu_feature = torch.cat((hidden, cell), dim=2)               # (num_layer=2, batchsize, 2*hidden_size=1024)
        qu_feature = qu_feature.transpose(0, 1)                     # (batchsize, num_layer=2, 2*hidden_size=1024)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)   # (batchsize, 2*num_layer*hidden_size=2048)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)                            # (batchsize, qu_feature_size=1024)

        return qu_feature

class LSTM_Encoder(nn.Module):

    def __init__(self, feature_size, qu_vocab_size, word_embed, hidden_size, num_hidden):

        super(LSTM_Encoder, self).__init__()
        self.img_encoder = Image_Encoder(feature_size)
        self.qu_encoder = Question_Encoder(qu_vocab_size, word_embed, hidden_size, num_hidden, feature_size)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(feature_size, 512)


    def forward(self, image, question):

        img_feature = self.img_encoder(image)               # (batchsize, feature_size=1024)
        qst_feature = self.qu_encoder(question)
        combined_feature = img_feature * qst_feature
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.fc(combined_feature)       # (batchsize, 512)

        return combined_feature