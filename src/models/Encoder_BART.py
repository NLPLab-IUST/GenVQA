import os
import random
import torch
from torch import nn

import torch.nn.functional as F
from transformers import ( 
    LxmertModel, 
    LxmertTokenizer, 
    VisualBertModel, 
    BertTokenizer,
    BartTokenizer)
from transformers.models.bart.modeling_bart import BartDecoder

class Encoder_BART(nn.Module):
    def __init__(self, encoder_type, bart_vesrion="facebook/bart-base", freeze_encoder=True):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'lxmert':
            self.encoder = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
            self.encoder.config.output_hidden_states = True
            self.encoder_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
            
        elif encoder_type == 'visualbert':
            self.encoder = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            self.encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        #freeze LXMERT
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        
        # This standard decoder layer is based on the paper “Attention Is All You Need”.
        self.Decoder = BartDecoder.from_pretrained(bart_vesrion).cuda()
        self.decoder_tokenizer = BartTokenizer.from_pretrained(bart_vesrion, bos_token='<cls>', cls_token='<cls>')
        self.Decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        
        self.output_size = self.decoder_tokenizer.vocab_size
        
        self.PADDING_VALUE = 0
        self.START_TOKEN = self.decoder_tokenizer.bos_token_id
        self.SEP_TOKEN = self.decoder_tokenizer.eos_token_id 
       
        self.Linear = nn.Linear(self.Decoder.config.d_model, self.decoder_tokenizer.vocab_size)

        self.name = f"{encoder_type}_bart"
        print(self.name)
    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized=None, teacher_force_ratio=0.5, max_seq_len=50):
        """
            Train phase forward propagation
        """
        batch_size = input_ids.shape[0]
        max_seq_len = max_seq_len if answer_tokenized is None else answer_tokenized.shape[0]

        # shift right
        answer_tokenized = answer_tokenized[:-1,:] if answer_tokenized is not None else answer_tokenized
        
        # encode question and image with lxmert
        if self.encoder_type == 'lxmert':
            kwargs = {"input_ids" : input_ids,
                      "visual_feats": visual_feats,
                      "visual_pos" : visual_pos,
                      "attention_mask": attention_mask}
            output = self.encoder(**kwargs)
            encoder_output  = output.language_hidden_states[-1]
            # encoder_output shape: (N, L, hidden_size) to send it to
            
            # memory masks to consider padding values in source sentence (questions)
            encoder_attention_mask = (input_ids == self.PADDING_VALUE).int()
        
        elif self.encoder_type == 'visualbert':
            kwargs = {"input_ids" : input_ids,
                      "attention_mask": attention_mask,
                      "visual_embeds": visual_feats,
                      "output_hidden_states":True}
            output = self.encoder(**kwargs)
            encoder_output = output.hidden_states[-1]
            # encoder_output shape: (N, L, hidden_size)
            
            # memory masks to consider padding values in source sentence (questions)
            encoder_attention_mask = (input_ids == self.PADDING_VALUE).int()
            encoder_attention_mask = F.pad(input=encoder_attention_mask, pad=(0, visual_feats.shape[1], 0, 0), mode='constant', value=0)
            # (batch_size, text_seq_length+image_seq_length)
        
        # if answer_tokenized is not None and random.random() < teacher_force_ratio:
        if answer_tokenized is not None:

            # target masks to consider padding values in target embeddings (answers)
            attention_mask = (answer_tokenized.permute(1, 0) == self.PADDING_VALUE).int()           

            # decode sentence and encoder output to generate answer
            output = self.Decoder(
                                input_ids=answer_tokenized.permute(1, 0), 
                                attention_mask=attention_mask,
                                encoder_hidden_states=encoder_output,
                                encoder_attention_mask = encoder_attention_mask,
                                output_hidden_states=True,
                                return_dict=True)
            # print(output.shape)
            output = output.last_hidden_state.permute(1, 0, 2)
            #output shape: (L, N, hidden_size)

            output = self.Linear(output)
            #output shape: (L, N, hidden_size)
        
        else:
            # generatoin phase
            x = torch.tensor([[self.START_TOKEN] * batch_size]).cuda()
            # x shape: (1, N)
            target_vocab_size = self.decoder_tokenizer.vocab_size

            outputs = torch.zeros(max_seq_len, batch_size, target_vocab_size).cuda()
            # outputs[0,:,self.START_TOKEN] = 1
            
            for i in range(0,max_seq_len):

                attention_mask = (x.permute(1, 0) == self.PADDING_VALUE).int()
                output = self.Decoder(
                                input_ids=x.permute(1, 0), 
                                attention_mask=attention_mask,
                                encoder_hidden_states=encoder_output,
                                encoder_attention_mask = encoder_attention_mask,
                                output_hidden_states=True,
                                return_dict=True)
                
                output = output.last_hidden_state.permute(1, 0, 2)
                #output shape: (L, N, hidden_size)
                
                # chose the last word of sequence Based on CodeXGLUE project
                # https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-refinement/code/model.py
                output = output[-1, :, :].unsqueeze(0)
                #output shape (1, N, hidden_size)
                

                output = self.Linear(output)
                #output shape: (1, N, vocab_size)
                
                outputs[i] = output
                
                #consider best guesses in a greeedy form! Better to implement with beam search
                output = torch.argmax(output, dim = -1)
                #output shape: (1, N)

                #concat new generated answer to x.
                x = torch.cat([x, output], dim=0)
                # x shape: (i + 1, N)
            
            output = outputs

        return output

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)

