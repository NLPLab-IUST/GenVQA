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
class VQA:
    def __init__(self, train_date, train_dset,  model, val_dset=None, tokenizer=None, use_cuda=True, batch_size=32, epochs=200, lr=0.005, log_every=1):
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, drop_last=True, collate_fn=pad_batched_sequence)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every
        if(use_cuda):
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.train_date_time = train_date
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        self.save_dir = os.path.join(CHECKPOINTS_DIR, str(self.train_date_time))
    def train(self):
        runnnin_loss = 0.0
        running_accuracy = 0.0
        for epoch in range(self.epochs):
            for i, (input_ids, feats, boxes, masks, target, target_masks) in enumerate(pbar := tqdm(self.train_loader, total=len(self.train_loader))):
                self.model.train()
                self.optim.zero_grad()
                pbar.set_description(f"Epoch {epoch}")
                target_masks = target_masks.unsqueeze(dim=2)
                logits = self.model(input_ids, feats, boxes, masks, target)
                target_labels = torch.nn.functional.one_hot(target, num_classes=self.model.Tokenizer.vocab_size).double()
                logits = logits * target_masks
                target_labels = target_labels*target_masks

                loss = self.criterion(logits, target_labels)
                loss.backward()
                self.optim.step()
                runnnin_loss += loss.item()

                #calculate accuracy
                pred = torch.argmax(logits, dim=-1)
                true_predictions = torch.sum((pred == target) * target_masks.squeeze())
                running_accuracy += true_predictions / (self.batch_size * torch.sum(target_masks))
                batch_acc = true_predictions / (self.batch_size * torch.sum(target_masks))
                assert batch_acc <= 1  
            if epoch % self.log_every == self.log_every - 1:
                total_data_iterated = self.log_every * len(self.train_loader)
                Logger.log("Train", f"Training epoch {epoch} with loss {runnnin_loss / self.log_every:.3f} with accuracy {running_accuracy / total_data_iterated:.3f}")
                self.model.save(self.save_dir, epoch)
                runnnin_loss = 0.0
                running_accuracy = 0.0
        
                





if __name__ == "__main__":
    model = LXMERT_LSTM.LXMERT_LSTM()
    dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_train/annotations.pickle", 
        questions = "../fsvqa_data_train/questions.pickle", 
        img_dir = "../img_data")
    vqa = VQA(datetime.now(), dset, model)
    vqa.train()