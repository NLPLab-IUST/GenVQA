import argparse
import os
from ast import arg
from datetime import datetime

import torch
import torch.nn as nn
from src.constants import CHECKPOINTS_DIR, LXMERT_HIDDEN_SIZE
from src.data.datasets import GenVQADataset, pad_batched_sequence
from src.logger import Instance as Logger
from src.models import LXMERT_RNN, LXMERT_Transformer, LXMERT_AttnRNN
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm


class VQA:
    def __init__(self,
                 train_date,
                 model,
                 train_dset,
                 val_dset=None,
                 tokenizer=None,
                 use_cuda=True,
                 batch_size=32,
                 epochs=200,
                 lr=0.005,
                 log_every=1,
                 save_every=50):
        
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every
        self.train_date_time = train_date
        self.save_every = save_every
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_batched_sequence)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_sequence)
        
        if(use_cuda):
            self.model = self.model.cuda()
            
        pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        
        self.f1_score = F1Score(num_classes=self.model.Tokenizer.vocab_size, ignore_index=pad_idx, top_k=1, mdmc_average='samplewise')
        self.accuracy = Accuracy(num_classes=self.model.Tokenizer.vocab_size, ignore_index=pad_idx, top_k=1, mdmc_average='samplewise')
        
        self.save_dir = os.path.join(CHECKPOINTS_DIR, str(self.train_date_time))
        
    def train(self):
        runnnin_loss = running_accuracy = running_accuracy_best = running_f1 = 0

        for epoch in range(self.epochs):
            for i, (input_ids, feats, boxes, masks, target, target_masks) in enumerate(pbar := tqdm(self.train_loader, total=len(self.train_loader))):

                self.model.train()
                
                pbar.set_description(f"Epoch {epoch}")
                loss, batch_acc, batch_f1 = self.__step(input_ids, feats, boxes, masks, target, target_masks, val=False)  
                
                runnnin_loss += loss.item()
                running_accuracy += batch_acc
                running_f1 += batch_f1
            
            if epoch % self.log_every == self.log_every - 1:
                val_loss = None
                val_acc = None
                if(self.val_loader):
                    self.model.eval()

                    val_loss = val_acc = val_f1 = 0
                    
                    for i, (input_ids, feats, boxes, masks, target, target_masks) in enumerate(self.val_loader):
                        val_loss, val_acc_batch, val_f1_batch = self.__step(input_ids, feats, boxes, masks, target, target_masks, val=True)
                        val_loss += loss.item()
                        val_acc += val_acc_batch
                        val_f1 += val_f1_batch
                    val_acc /= len(self.val_loader)
                    val_f1 /= len(self.val_loader)

                total_data_iterated = self.log_every * len(self.train_loader)
                running_accuracy = running_accuracy / total_data_iterated
                running_f1 /= total_data_iterated

                
                if(self.val_loader):
                    Logger.log(f"Train_{self.train_date_time}", f"Training epoch {epoch}: Train loss {runnnin_loss / self.log_every:.3f}. Val loss: {val_loss:.3f}."
                                + f" Train accuracy {running_accuracy:.3f}. Val accuracy: {val_acc:.3f}. Train F1-Score: {running_f1}. Validation F1-Score: {val_f1}")
                    print(f"F1 Score: Train {running_f1}, Validation: {val_f1}")
                else:
                    Logger.log(f"Train_{self.train_date_time}", f"Training epoch {epoch}: Train loss {runnnin_loss / self.log_every:.3f}."
                                + f" Train accuracy {running_accuracy:.3f}. Train F1-Score: {running_f1}")
                    print(f"F1 Score: Train {running_f1}")

                
                if(running_accuracy > running_accuracy_best):
                    self.model.save(self.save_dir, f"BEST")
                    running_accuracy_best = running_accuracy
                
                runnnin_loss = running_accuracy = running_f1 = 0

            if(epoch % self.save_every == self.save_every - 1):
                self.model.save(self.save_dir, epoch)
            
    
    def __step(self, input_ids, feats, boxes, masks, target, target_masks, val=False):        
        logits = self.model(input_ids, feats, boxes, masks, target)
        # logits shape: (L, N, target_vocab_size)
        loss = self.criterion(logits.permute(1, 2, 0), target.permute(1,0))

        if not(val):
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        f1_score = self.f1_score(logits.permute(1,2,0), target.permute(1,0))
        batch_acc = self.accuracy(logits.permute(1,2,0), target.permute(1,0))
        # pred = torch.argmax(logits, dim=-1)
        # true_predictions = torch.sum((pred == target) * target_masks)   
        # batch_acc = true_predictions / (self.batch_size * torch.sum(target_masks))
        
        assert batch_acc <= 1
        return loss, batch_acc, f1_score
                

def parse_args():
    parser = argparse.ArgumentParser()
    
    #specify decoder type, options: rnn, attn-rnn, transformer, 
    parser.add_argument("--decoder_type", default="rnn", type=str)

    #RNN specifications
    parser.add_argument("--rnn_type", default="lstm", type=str) #options: lstm, gru
    parser.add_argument("--num_rnn_layers", default=1, type=int)
    parser.add_argument("--bidirectional", default=False, action="store_true")
    
    # Attention RNN specifications
    parser.add_argument("--attn_type", default="bahdanau", type=str) #options: bahdanau, luong
    # use only when attention type is luong
    parser.add_argument("--attn_method", default="dot", type=str) #options: dot, general, concat
    
    #Transformer specifications
    parser.add_argument("--nheads", default=12, type=int)
    parser.add_argument("--num_transformer_layers", default=6, type=int)

    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    model = None
    if (args.decoder_type.lower() == 'rnn'):
        model = LXMERT_RNN.LXMERT_RNN(rnn_type=args.rnn_type, 
                                    num_layers=args.num_rnn_layers, 
                                    bidirectional=args.bidirectional)
    
    elif (args.decoder_type.lower() == 'transformer'):
        model = LXMERT_Transformer.LXMERT_Transformer(
                                    args.nheads, 
                                    args.num_transformer_layers, 
                                    LXMERT_HIDDEN_SIZE).cuda()
        
    elif(args.decoder_type.lower() == 'attn-rnn'):
        model = LXMERT_AttnRNN.LXMERT_AttnRNN(rnn_type=args.rnn_type,
                                              attn_type = args.attn_type,
                                              attn_method=args.attn_method)
                                   
    train_dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_train/annotations.pickle", 
        questions = "../fsvqa_data_train/questions.pickle", 
        img_dir = "../img_data")
    val_dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_val/annotations.pickle", 
        questions = "../fsvqa_data_val/questions.pickle", 
        img_dir = "../val_img_data")
    
    if model:
        vqa = VQA(datetime.now() ,model, train_dset, val_dset=val_dset)
        vqa.train()
