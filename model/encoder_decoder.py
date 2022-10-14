
from model.decoder import LSTM_Decoder
from model.encoder import LSTM_Encoder
from build_dataset.build_dataset import data_loader

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




BATCH_SIZE = 150
MAX_QU_LEN = 30
NUM_WORKER = 8
FEATURE_SIZE, WORD_EMBED = 1024, 300
NUM_HIDDEN, HIDDEN_SIZE = 2, 512
LEARNING_RATE, STEP_SIZE, GAMMA = 0.001, 10, 0.1
EPOCH = 50
DATA_DIR = '../data'

dataloader = data_loader(input_dir=DATA_DIR, batch_size=BATCH_SIZE, max_qu_len=MAX_QU_LEN, num_worker=NUM_WORKER)
qu_vocab_size = dataloader['train'].dataset.qu_vocab.vocab_size

class LSTM_Encoder_Decoder(torch.nn.Module):
    def __init__(self, num_layers=2, bidirectional=False, prob=0.5, freeze_encoder=True):
        super().__init__()
      
        self.encoder = LSTM_Encoder(feature_size=FEATURE_SIZE, qu_vocab_size=qu_vocab_size, word_embed=WORD_EMBED, 
                                    hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN)
        #freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        self.embedding_layer = nn.Embedding(qu_vocab_size=qu_vocab_size, word_embed=WORD_EMBED)

        self.decoder = LSTM_Decoder(embedding=self.embedding_layer,
                            output_size=qu_vocab_size, 
                            num_layers=2, 
                            bidirectional=False, 
                            prob=0.5)
        
        self.start_token = 101 # <cls>
        self.end_token = 102 # <sep>
        
        self.D = 2 if bidirectional==True else 1

    def forward(self, image, question, answer_tokenized = None, teacher_force_ratio=0.5, max_sequence_length=50):
        """
            Train phase forward propagation
        """
        
        batch_size = 150
        target_len = max_sequence_length if answer_tokenized is None else answer_tokenized.shape[0]
        target_vocab_size = self.Tokenizer.vocab_size
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).cuda()
        
        #encoder
  
        encoder_output = self.encoder(image, question)
            
        # encoder_output shape: (N, hidden_size) to send it to Decoder as hidden,
        # we want it to be (D*num_layers, N, hidden_size) so we're just gonna expand it.
        
        h = encoder_output.expand(self.D*self.decoder.num_layers, -1, -1)
        # h shape: (D*num_layers, N, hidden_size)
        
        c = torch.zeros(*h.shape).cuda()
        hidden = (h.contiguous(),c.contiguous())
            
        # Send <cls> token to decoder
        x =  torch.tensor([self.start_token]*batch_size).cuda()
        
        for t in range(0, target_len):
            output, hidden = self.decoder(x, hidden)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to.
            x = answer_tokenized[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
