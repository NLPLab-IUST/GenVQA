import torch

class GreedyDecoder():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def decode_from_logits(self, logits):
        logits = torch.argmax(logits, dim=-1)    
        return self.batch_decode(logits)

    def batch_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)