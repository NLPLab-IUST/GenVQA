from transformers import LxmertTokenizer, LxmertModel
import torch
from src.models.LSTM import LSTMModel
import os
from datetime import datetime
class LXMERT_LSTM(torch.nn.Module):
    def __init__(self, freeze_lxmert=True):
        super().__init__()
        self.LXMERT = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        
        #freeze LXMERT
        if freeze_lxmert:
            for p in self.LXMERT.parameters():
                p.requires_grad = False
        
        self.embedding_layer = list(self.LXMERT.children())[0].word_embeddings
        self.LSTM = LSTMModel(output_size=self.Tokenizer.vocab_size)
        self.name = "LXMERT_LSTM"
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
        answer_embeddings = self.embedding_layer(answer_tokenized)
        output = self.LXMERT(**kwargs)
        output = output.pooled_output
        output = self.LSTM(answer_embeddings, output)
        return output
    def save(self, dir_, epoch):
        dir_ = os.path.join(dir_, str(datetime().now()))
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
