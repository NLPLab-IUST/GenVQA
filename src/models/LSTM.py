
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size = 768, hidden_size=768, output_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.LSTM = LSTM(input_size=input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=False, num_layers=1)
        self.Linear = nn.Linear(self.hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, encoder_output, x_lens):
        """
            Predicts the values based on the true labels. This function is proper for 
            training phase. You have to use predict() for testing or interence phase.

            x : Input Tensor of size (L, N, embedding_size)
            encoder_output: encode output of size (N, hidden_size)
        """
        h0 = encoder_output.unsqueeze(0)
        c0 = torch.zeros(*h0.shape).cuda()
        x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(x, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.Linear(output)
        return self.log_softmax(output)