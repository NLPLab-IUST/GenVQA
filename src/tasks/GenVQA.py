import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.models import LXMERT_LSTM
from src.data.datasets import GenVQADataset, pad_batched_sequence
import torch

class VQA:
    def __init__(self, train_dset,  model, val_dset=None, tokenizer=None, use_cuda=True, batch_size=32, epochs=50, lr=5e-5):
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True, collate_fn=pad_batched_sequence)
        self.model = model
        self.epochs = epochs
        if(use_cuda):
            self.model = self.model.cuda()
        self.loss = nn.BCEWithLogitsLoss()

        batch_per_epoch = len(self.train_loader)
        t_total = int(batch_per_epoch * epochs)
        print("BertAdam Total Iters: %d" % t_total)
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
    def train(self):
        for i, (input_ids, feats, boxes, masks, target) in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
            print(input_ids.shape)
            print(boxes.shape)
            print(feats.shape)
            print(masks.shape)
            print(target.shape)
            self.model.train()
            self.optim.zero_grad()


if __name__ == "__main__":
    model = LXMERT_LSTM.LXMERT_LSTM()
    dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_train/annotations.pickle", 
        questions = "../fsvqa_data_train/questions.pickle", 
        img_dir = "../img_data")
    vqa = VQA(dset, model)
    vqa.train()