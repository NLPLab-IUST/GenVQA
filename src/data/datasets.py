import pickle
import os
from torch.utils.data import Dataset

class GenVQADataset(Dataset):
    def __init__(self, embedding_layer, tokenizer, annotations, questions, img_dir):
        with open(annotations, 'rb') as f:
            self.annotations = pickle.load(f)
        with open(questions, 'rb') as f:
            self.questions = pickle.load(f)
        self.img_dir = img_dir
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        dataum = self.annotations[idx]
        q = self.questions[dataum['question_id']]
        img_path = os.path.join(self.img_dir, dataum['img_id'])
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        tokenized_sentence = self.tokenizer(q['question'])
        input_ids = tokenized_sentence['input_ids']
        attention_mask = tokenized_sentence['attention_mask']
        visual_feats = img['features']
        boxes = img['boxes']
        img_h, img_w = img_features['img_h'], img_features['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h

        # labels
        if 'answers' in dataum.keys():
            answer = dataum['answers'][0]
            a_text = answer['answer']
            tokenized_sentence = self.tokenizer(a_text)
            label_tokenized = tokenized_sentence['input_ids']
            return input_ids, visual_feats, visual_pos, attention_mask, label_tokenized
        
        return input_ids, visual_feats, visual_pos, attention_mask, None