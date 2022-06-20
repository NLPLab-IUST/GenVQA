import os
import random

import torch
from transformers import LxmertModel, LxmertTokenizer

from models.RNN import RNNModel


class LXMERT_RNN(torch.nn.Module):
    def __init__(self, rnn_type = "lstm", num_layers=1, bidirectional=False, p=0.5, freeze_lxmert=True):
        super().__init__()
        self.LXMERT = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        #freeze LXMERT
        if freeze_lxmert:
            for p in self.LXMERT.parameters():
                p.requires_grad = False
        
        embedding_layer = list(self.LXMERT.children())[0].word_embeddings
        self.rnn = RNNModel(embedding=embedding_layer,
                             output_size=self.Tokenizer.vocab_size,
                             rnn_type=rnn_type,
                             num_layers=num_layers,
                             bidirectional=bidirectional,
                             p=p)
        
        self.name = f"LXMERT_{rnn_type}_{num_layers}"
        self.name = f"{self.name}_bidirectional" if bidirectional else self.name

    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized, teacher_force_ratio=0.5):
        """
            Train phase forward propagation
        """
        kwargs = {
            "input_ids" : input_ids,
            "visual_feats": visual_feats,
            "visual_pos" : visual_pos,
            "attention_mask": attention_mask
        }
        
        batch_size = answer_tokenized.shape[1]
        target_len = answer_tokenized.shape[0]
        target_vocab_size = self.Tokenizer.vocab_size
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).cuda()
        
        #encoder
        output = self.LXMERT(**kwargs)
        encoder_output = output.pooled_output
        # encoder_output shape: (N, hidden_size) to send it to
        # Decoder as hidden we want it to be (D*num_layers, N, hidden_size) so we're
        # just gonna expand it.
        # D = 2 if bidirectional=True otherwise 1
        
        D = 2 if self.rnn.bidirectional==True else 1
        
        hidden = encoder_output.expand(D*self.rnn.num_layers, -1, -1)
        # hidden shape: (D*num_layers, N, hidden_size)
        cell = torch.zeros(*hidden.shape).cuda()
        
        # Grab the first input to the Decoder which will be <SOS> token
        x = answer_tokenized[0]
        
        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.rnn(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = answer_tokenized[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
