import torch
import torch.nn as nn
from torch.nn import GRU, LSTM


class RNNModel(torch.nn.Module):
    def __init__(self, embedding, rnn_type="lstm", input_size=768, hidden_size=768, output_size=768, num_layers=1, bidirectional=False, p=0.5):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.dropout = nn.Dropout(p)
        if rnn_type=="lstm":
            self.rnn = LSTM(input_size=input_size, hidden_size=self.hidden_size, 
                            bidirectional=self.bidirectional, 
                            num_layers=self.num_layers, dropout = p)
        elif rnn_type=="gru":
            self.rnn = GRU(input_size=input_size, hidden_size=self.hidden_size, 
                            bidirectional=self.bidirectional, 
                            num_layers=self.num_layers, dropout = p)
        self.Linear = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        """
            # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
            # is 1 here because we are sending in a single word and not a sentence
        """
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)
        
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)
        
        predictions = self.Linear(outputs)
        
        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
