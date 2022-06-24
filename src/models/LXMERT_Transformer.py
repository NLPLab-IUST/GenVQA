import torch
from torch import nn

from transformers import LxmertModel, LxmertTokenizer
class LXMERT_Transformer(nn.Module):
    def __init__(self, nheads, decoder_layers, hidden_size, freeze_lxmert=True):
        super().__init__()
        self.LXMERT = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.LXMERT.config.output_hidden_states = True
        self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        #freeze LXMERT
        if freeze_lxmert:
            for p in self.LXMERT.parameters():
                p.requires_grad = False

        
        # This standard decoder layer is based on the paper “Attention Is All You Need”.
        transformer_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nheads)
        self.Decoder = nn.TransformerDecoder(transformer_layer, num_layers=decoder_layers)

        self.embedding_layer = self.LXMERT.embeddings.word_embeddings
        self.output_size = self.Tokenizer.vocab_size
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.hidden_size = hidden_size
        self.PADDING_VALUE = 0
        
        #Linear layer to output vocabulary size
        self.Linear = nn.Linear(hidden_size, self.Tokenizer.vocab_size)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized):
        """
            Train phase forward propagation
        """
        kwargs = {
            "input_ids" : input_ids,
            "visual_feats": visual_feats,
            "visual_pos" : visual_pos,
            "attention_mask": attention_mask
        }
        
        tgt_len = answer_tokenized.shape[0]
        
        #encoder
        output = self.LXMERT(**kwargs)
        encoder_output = output.language_hidden_states[-1].permute(1, 0, 2)
        # encoder_output shape: (seq_len, N, hidden_size) to send it to
        
        
        answer_embeddings = self.embedding_layer(answer_tokenized)
        
        #create mask tensor to ignore paddings during calcualtion
        tgt_key_padding_mask = (answer_tokenized.permute(1, 0) == 0)
        memory_key_padding_mask = (input_ids == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()        
        
        #decode encoder output and generate output sentence
        output = self.Decoder( answer_embeddings, 
                               encoder_output,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask, 
                               memory_key_padding_mask = memory_key_padding_mask)
        #output shape: (tgt_seq_len, N, hidden_size)
        
        output = self.Linear(output)
        #output shape: (tgt_seq_len, N, vocab_size)

        return self.LogSoftmax(output)



