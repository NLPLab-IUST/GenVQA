import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.models import LXMERT_LSTM
from src.data.datasets import GenVQADataset
class VQA:
    def __init__(self, train_dset,  model, val_dset=None, tokenizer=None, use_cuda=True, batch_size=32, epochs=50, lr=5e-5):
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        self.model = model
        self.epochs = epochs
        if(use_cuda):
            self.model = self.model.cuda()
        self.loss = nn.BCEWithLogitsLoss()

        batch_per_epoch = len(train_loader)
        t_total = int(batch_per_epoch * epochs)
        print("BertAdam Total Iters: %d" % t_total)
        from lxrt.optimization import BertAdam
        self.optim = BertAdam(list(self.model.parameters()),
                                lr=lr,
                                warmup=0.1,
                                t_total=t_total)
    def train(self):
        for i, (input_ids, feats, boxes, masks, target) in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
            self.model.train()
            self.optim.zero_grad()


if __name__ == "__main__":
    model = LXMERT_LSTM()
    dset = GenVQADataset(model.Tokenizer, 
        annotations = "../fsvqa_data_trian/annotations.pickle", 
        questions = "../fsvqa_data_trian/questions.pickle", 
        img_dir = "../img_data")
    vqa = VQA(dset, model)
    vqa.train()