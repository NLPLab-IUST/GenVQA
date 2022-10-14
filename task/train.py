# !pip install torchmetrics
# !pip install transformers

import argparse
from cgi import test
import json
import os
import random
from datetime import datetime
from build_dataset.build_dataset import data_loader
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from transformers import AdamW
import torch.nn.functional as F

BASE_DIR = "./"
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
class VQA:
    def __init__(self,
                 train_date,
                 model,
                 decoder_type,
                 train_dset,
                 val_dset=None,
                 test_dset=None,
                 use_cuda=True,
                 batch_size=32,
                 epochs=200,
                 lr=0.005,
                 log_every=1,
                 save_every=5, 
                 max_sequence_length=50, 
                 optimizer = 'adam'):
        
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every
        self.train_date_time = train_date
        self.save_every = save_every
        self.decoder_type = decoder_type
        self.max_sequence_length = max_sequence_length
        
        self.train_loader = data_loader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_batched_sequence)
        self.val_loader = data_loader(val_dset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_sequence)

        if(use_cuda):
            self.model = self.model.cuda()
            
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        if optimizer == 'adam':
            self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        elif optimizer =='sgd':
            self.optim = torch.optim.SGD(list(self.model.parameters()), lr=lr)
        elif optimizer =='adamw':
            self.optim = AdamW(list(self.model.parameters()), lr=lr)
            
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.5)
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        
        self.f1_score = F1Score(num_classes=self.model.Tokenizer.vocab_size, ignore_index=self.pad_idx, top_k=1, mdmc_average='samplewise')
        self.accuracy = Accuracy(num_classes=self.model.Tokenizer.vocab_size, ignore_index=self.pad_idx, top_k=1, mdmc_average='samplewise')
        
        self.save_dir = os.path.join(CHECKPOINTS_DIR, str(self.train_date_time))
        if not(os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        running_loss = running_accuracy = running_accuracy_best = running_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            for i, (input_ids, feats, boxes, masks, target) in enumerate(pbar := tqdm(self.train_loader, total=len(self.train_loader))):
                # torch.cuda.empty_cache()
                pbar.set_description(f"Epoch {epoch}")
                loss, batch_acc, batch_f1, _ = self.__step(input_ids, feats, boxes, masks, target, val=False)  
                
                running_loss += loss.item()
                running_accuracy += batch_acc.item()
                running_f1 += batch_f1
                pbar.set_postfix(loss=running_loss/(i+1), accuracy=running_accuracy/(i+1))

            if epoch % self.log_every == self.log_every - 1:                                
                val_loss, val_acc, val_f1, _ = self.__evaluate_validation()
                
                total_data_iterated = self.log_every * len(self.train_loader)
                running_loss /= total_data_iterated
                running_accuracy /= total_data_iterated
                running_f1 /= total_data_iterated
                
                #logging results
                Logger.log(f"Train_{self.train_date_time}", f"Training epoch {epoch}: Train loss {running_loss:.3f}. Val loss: {val_loss:.3f}."
                            + f" Train accuracy {running_accuracy:.3f}. Val accuracy: {val_acc:.3f}. Train F1-Score: {running_f1}. Validation F1-Score: {val_f1}")
                print(f"F1 Score: Train {running_f1}, Validation: {val_f1}")

                if(running_accuracy > running_accuracy_best):
                    self.model.save(self.save_dir, "BEST")
                    running_accuracy_best = running_accuracy
                
                running_loss = running_accuracy = running_f1 = 0
            
            if(epoch % self.save_every == self.save_every - 1):
                self.model.save(self.save_dir, epoch)

            # self.scheduler.step()    
            
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            
    @torch.no_grad()   
    def __evaluate_validation(self, metric_calculator=False, dset=None):
        print("Validation Evaluations: ")
        self.model.eval()
        val_loss = val_acc = val_f1 = 0

        if(dset):
            loader = data_loader(dset, batch_size=self.batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_sequence)
        else:
            loader = self.val_loader
        # define metric calculator if we need extra metric calculation
        if(metric_calculator):
            metric_calculator = MetricCalculator(self.model.embedding_layer)
            # we used greedy decoder as a temporary decode. 
            decoder = GreedyDecoder(self.model.Tokenizer)

            
        
        for i, (image, question, target) in enumerate(pbar := tqdm(loader, total=len(loader))):
            #calculate losses, and logits + necessary metrics for showin during training
            # torch.cuda.empty_cache()
            loss, val_acc_batch, val_f1_batch, logits = self.__step(image, question, target, val=True)
            
            val_loss += loss.item()
            val_acc += val_acc_batch.item()
            val_f1 += val_f1_batch
            pbar.set_postfix(loss=val_loss/(i+1), accuracy=val_acc/(i+1))
            
            #only when we need extra metrics for evaluation!
            if(metric_calculator):
                # using argmax to find the best token!
                preds_tokenized = decoder.decode_from_logits(logits)

                #tokenized sentences without [PAD] and [SEP] tokens. pure sentences!
                pred_sentences_decoded, preds_sentences_ids = decoder.batch_decode(preds_tokenized.permute(1, 0))
                ref_sentences_decoded, ref_sentences_ids = decoder.batch_decode(target.permute(1, 0))
                
                #calculate metrics such as BLEU, ROUGE, BERTSCORE, and others.
                #it accumalates values to be calculated later
                metric_calculator.add_batch(pred_sentences_decoded, ref_sentences_decoded, preds_sentences_ids, ref_sentences_ids)

        val_loss /= len(loader)
        val_acc /= len(loader)
        val_f1 /= len(loader)
        
        #calculate metrics based on the accumelated metrics during evaluation!
        other_metrics = metric_calculator.compute() if metric_calculator else None
        
        return val_loss, val_acc, val_f1, other_metrics
        
    def __step(self, image, question, target, val=False):
        
        teacher_force_ratio = 0 if val else 0.5
        answer_tokenized = None if val else target      
        logits = self.model(image, question, answer_tokenized, teacher_force_ratio, self.max_sequence_length)
        
        # logits shape: (L, N, target_vocab_size)

        if val:
            target = F.pad(input=target, pad=(0, 0, 0, self.max_sequence_length - target.shape[0]), mode='constant', value=self.pad_idx)
            
        loss = self.criterion(logits.permute(1, 2, 0), target.permute(1,0))

        if not(val):
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()

        f1_score = self.f1_score(logits.permute(1,2,0), target.permute(1,0))
        batch_acc = self.accuracy(logits.permute(1,2,0), target.permute(1,0))

        return loss, batch_acc, f1_score, logits

    def evaluate(self, dset, key):
        _ , val_acc, val_f1, other_metrics = self.__evaluate_validation(metric_calculator=True, dset= dset)
        other_metrics["accuracy"] = val_acc
        other_metrics['f1'] = val_f1.cpu().tolist()
        with open(os.path.join(self.save_dir, f"evaluation_{key}.json"), 'w') as fp:
            json.dump(other_metrics, fp)
    
    def load_model(self, key):
        path = os.path.join(self.save_dir, f"{self.model.name}.{key}.torch")
        if not (os.path.exists(path)):
            Logger.log(f"Train_{self.train_date_time}", f"Couldn't load model from {path} ")
            return

        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        
    def predict(self, model_path, dset, key):
        #load model
        if os.path.exists(model_path) == False:
            print(f"Couldn't load model from {model_path}")
            return

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        
        # load dataset
        if(dset):
            loader = data_loader(dset, batch_size=self.batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_sequence)
        else:
            loader = self.val_loader
            
        self.model.eval()
        decoder = GreedyDecoder(self.model.Tokenizer)
        questions, pred_sentences, ref_sentences = [], [], []
        
        for i, (image, question, target) in enumerate(pbar := tqdm(loader, total=len(loader))):
            _, _, _, logits = self.__step(image, question, target, val=True)
            
            with torch.no_grad():
                logits = self.model(image, question)
                predict = torch.log_softmax(logits, dim=1)

            predict = torch.argmax(predict, dim=1).tolist()
            predict = [loader.ans_vocab.idx2word(idx) for idx in predict]
            ans_qu_pair = [{'answer': ans, 'question_id': id} for ans, id in zip(predict, question_id)]

            preds_tokenized = decoder.decode_from_logits(logits)
            questions_decoded, _ = decoder.batch_decode(input_ids)
            pred_sentences_decoded, _ = decoder.batch_decode(preds_tokenized.permute(1, 0))
            ref_sentences_decoded, _ = decoder.batch_decode(target.permute(1, 0))
            
            questions.extend(questions_decoded)
            pred_sentences.extend(pred_sentences_decoded)
            ref_sentences.extend(ref_sentences_decoded)
            
            
        model_predictions = [{"question":question, "ref answer": ref_answer, "pred answer":pred_answer} 
                             for question, ref_answer, pred_answer in zip(questions, ref_sentences, pred_sentences)]
              
        with open(os.path.join(os.path.split(model_path)[0], f"model_prediction_{key}.json"), 'w') as fp:
            json.dump(model_predictions, fp)