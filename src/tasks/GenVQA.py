import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.models import LXMERT_LSTM
from src.data.datasets import GenVQADataset, pad_batched_sequence
import torch
from src.logger import Instance as Logger
class VQA:
    def __init__(self, train_dset,  model, val_dset=None, tokenizer=None, use_cuda=True, batch_size=32, epochs=200, lr=0.005):
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, drop_last=True, collate_fn=pad_batched_sequence)
        self.model = model
        self.epochs = epochs
        if(use_cuda):
            self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
    def train(self):
        runnnin_loss = 0.0
        for epoch in range(self.epochs):
            for i, (input_ids, feats, boxes, masks, target) in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                self.model.train()
                self.optim.zero_grad()
                logits = self.model(input_ids, feats, boxes, masks, target)
                target_labels = torch.nn.functional.one_hot(target, num_classes=self.model.Tokenizer.vocab_size).double()
                loss = self.criterion(logits, target_labels)
                loss.backward()
                self.optim.step()
                runnnin_loss += loss.item()
            if epoch % 1 == 0:
                Logger.log("Train", f"Training epoch {epoch} with loss {runnnin_loss / 1:.3f}")
                runnnin_loss = 0.0
        
                





if __name__ == "__main__":
    model = LXMERT_LSTM.LXMERT_LSTM()
    dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_train/annotations.pickle", 
        questions = "../fsvqa_data_train/questions.pickle", 
        img_dir = "../img_data")
    vqa = VQA(dset, model)
    vqa.train()