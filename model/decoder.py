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



class LSTM_Decoder(nn.Module):
    def __init__(self, embedding, input_size=512, hidden_size=512, output_size=512, num_layers=2, bidirectional=False, prob=0.5):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.dropout = nn.Dropout(prob)
        
        dropout = 0 if self.num_layers == 1 else prob
        
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size, 
                        bidirectional=self.bidirectional, 
                        num_layers=self.num_layers, dropout = dropout)
            
        D = 2 if self.lstm.bidirectional==True else 1
        self.Linear = nn.Linear(D*self.hidden_size, output_size)
    
    def forward(self, x, hidden):
        """
            # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
            # is 1 here because we are sending in a single word and not a sentence
        """
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)
        
        outputs, hidden = self.lstm(embedding, hidden)
        # outputs shape: (1, N, hidden_size)
        
        predictions = self.Linear(outputs)
        
        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden
