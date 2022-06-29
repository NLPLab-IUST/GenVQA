import os

import torch
from torch import nn

import torch.nn.functional as F
from transformers import LxmertModel, LxmertTokenizer, VisualBertModel, BertTokenizer

class Encoder_Transformer(nn.Module):
    def __init__(self, encoder_type, nheads, decoder_layers, hidden_size, freeze_encoder=True):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'lxmert':
            self.encoder = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
            self.encoder.config.output_hidden_states = True
            self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
            
        elif encoder_type == 'visualbert':
            self.encoder = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            self.Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        #freeze LXMERT
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        
        # This standard decoder layer is based on the paper “Attention Is All You Need”.
        transformer_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nheads)
        self.Decoder = nn.TransformerDecoder(transformer_layer, num_layers=decoder_layers)

        self.embedding_layer = self.encoder.embeddings.word_embeddings
        self.output_size = self.Tokenizer.vocab_size
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.hidden_size = hidden_size
        self.PADDING_VALUE = 0
        
        #Linear layer to output vocabulary size
        self.Linear = nn.Linear(hidden_size, self.Tokenizer.vocab_size)
        
        self.name = f"{encoder_type}_{nheads}heads_{decoder_layers}_transformer"
        print(self.name)
    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized):
        """
            Train phase forward propagation
        """
        
        tgt_len = answer_tokenized.shape[0]
        
        # encode question and image with lxmert
        if self.encoder_type == 'lxmert':
            kwargs = {"input_ids" : input_ids,
                      "visual_feats": visual_feats,
                      "visual_pos" : visual_pos,
                      "attention_mask": attention_mask}
            output = self.encoder(**kwargs)
            encoder_output  = output.language_hidden_states[-1].permute(1, 0, 2)
            # encoder_output shape: (seq_len, N, hidden_size) to send it to
            
            # memory masks to consider padding values in source sentence (questions)
            memory_key_padding_mask = (input_ids == self.PADDING_VALUE)
        
        elif self.encoder_type == 'visualbert':
            kwargs = {"input_ids" : input_ids,
                      "attention_mask": attention_mask,
                      "visual_embeds": visual_feats,
                      "output_hidden_states":True}
            output = self.encoder(**kwargs)
            encoder_output = output.hidden_states[-1].permute(1,0,2)
            # encoder_output shape: (sequence_length, batch_size, hidden_size)
            
            # memory masks to consider padding values in source sentence (questions)
            memory_key_padding_mask = (input_ids == self.PADDING_VALUE)
            memory_key_padding_mask = F.pad(input=memory_key_padding_mask, pad=(0, visual_feats.shape[1], 0, 0), mode='constant', value=0)
            # (batch_size, text_seq_length+image_seq_length)
        
        answer_embeddings = self.embedding_layer(answer_tokenized)
        # answer embeddings shape: (seq_len, N, embedding_size)
        # embedding_size is 768 in LXMERT

        # target masks to consider padding values in target embeddings (answers)
        tgt_key_padding_mask = (answer_tokenized.permute(1, 0) == self.PADDING_VALUE)

        # target attention masks to avoid future tokens in our predictions
        # Adapted from PyTorch source code:
        # https://github.com/pytorch/pytorch/blob/176174a68ba2d36b9a5aaef0943421682ecc66d4/torch/nn/modules/transformer.py#L130
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()        
        

        # decode sentence and lxmert output to generate answer
        output = self.Decoder( answer_embeddings, 
                               encoder_output,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask, 
                               memory_key_padding_mask = memory_key_padding_mask)
        #output shape: (tgt_seq_len, N, hidden_size)
        
        output = self.Linear(output)
        #output shape: (tgt_seq_len, N, vocab_size)

        return output

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)

