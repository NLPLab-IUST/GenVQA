import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.models import LXMERT_LSTM
from src.data.datasets import GenVQADataset, pad_batched_sequence
import torch
from src.logger import Instance as Logger
from src.constants import CHECKPOINTS_DIR
from datetime import datetime
import os
from torchmetrics import F1Score
class VQA:
    def __init__(self, train_date, train_dset,  model, 
                val_dset=None, 
                tokenizer=None, 
                use_cuda=True, 
                batch_size=32, 
                epochs=200, 
                lr=0.005, 
                log_every=1, 
                save_every=50):
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, drop_last=True, collate_fn=pad_batched_sequence)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every
        self.val_loader = None
        if(val_dset):
            self.val_loader = DataLoader(val_dset, batch_size=batch_size,
            shuffle=False, drop_last=True, collate_fn=pad_batched_sequence)
        if(use_cuda):
            self.model = self.model.cuda()
        self.criterion = nn.NLLLoss(ignore_index=0, size_average=True)
        self.train_date_time = train_date
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        self.save_dir = os.path.join(CHECKPOINTS_DIR, str(self.train_date_time))
        self.save_every = save_every
        self.f1_score = F1Score(num_classes=self.model.Tokenizer.vocab_size, ignore_index=0, top_k=1)
    def train(self):
        runnnin_loss = 0.0
        running_accuracy = 0.0
        running_accuracy_best = 0
        running_f1 = 0
        for epoch in range(self.epochs):
            for i, (input_ids, feats, boxes, masks, target, target_masks) in enumerate(pbar := tqdm(self.train_loader, total=len(self.train_loader))):
                self.model.train()
                self.optim.zero_grad()
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
                    val_loss = 0
                    val_acc = 0
                    val_f1 = 0
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
                    Logger.log("Train", f"Training epoch {epoch}: Train loss {runnnin_loss / self.log_every:.3f}. Val loss: {val_loss:.3f}."
                                + f" Train accuracy {running_accuracy:.3f}. Val accuracy: {val_acc:.3f}. Train F1-Score: {running_f1}. Validation F1-Score: {val_f1}")
                    print(f"F1 Score: Train {running_f1}, Validation: {val_f1}")
                else:
                    Logger.log("Train", f"Training epoch {epoch}: Train loss {runnnin_loss / self.log_every:.3f}."
                                + f" Train accuracy {running_accuracy:.3f}. Train F1-Score: {running_f1}")
                    print(f"F1 Score: Train {running_f1}")

                
                if(running_accuracy > running_accuracy_best):
                    self.model.save(self.save_dir, f"BEST")
                    running_accuracy_best = running_accuracy
                runnnin_loss = 0.0
                running_accuracy = 0.0
                running_f1 = 0
            if(epoch % self.save_every == self.save_every - 1):
                self.model.save(self.save_dir, epoch)
            
    
    def __step(self, input_ids, feats, boxes, masks, target, target_masks, val=False):
        target_lens = torch.sum(target_masks, dim=1)
        logits = self.model(input_ids, feats, boxes, masks, target, target_lens)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.model.Tokenizer.vocab_size).double()
        loss = self.criterion(logits.permute(0, 2, 1), target)
        if not(val):
            loss.backward()
            self.optim.step()
        f1_score = self.f1_score(logits.permute(0, 2, 1), target)
        pred = torch.argmax(logits, dim=-1)
        true_predictions = torch.sum((pred == target) * target_masks)
        batch_acc = true_predictions / (self.batch_size * torch.sum(target_masks))
        assert batch_acc <= 1
        return loss, batch_acc, f1_score
                





if __name__ == "__main__":
    model = LXMERT_LSTM.LXMERT_LSTM()
    train_dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_train/annotations.pickle", 
        questions = "../fsvqa_data_train/questions.pickle", 
        img_dir = "../img_data")
    val_dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_val/annotations.pickle", 
        questions = "../fsvqa_data_val/questions.pickle", 
        img_dir = "../val_img_data")
    vqa = VQA(datetime.now(), train_dset, model, val_dset=val_dset)
    vqa.train()