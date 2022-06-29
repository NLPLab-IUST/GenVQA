import torch
import evaluate
from src.metrics.EmbeddingBase.AverageScore import AverageScore
from src.metrics.EmbeddingBase.ExtremaScore import ExtremaScore
from src.metrics.EmbeddingBase.GreedyMatchingScore import GreedyMatchingScore
from src.metrics.cider.cider import Cider
#!pip install evaluate
#!pip install rouge_score
#!pip install bert_score

class MetricCalculator():
    def __init__(self, tokenizer, embedding_layer) -> None:
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.METRICS = ["average_score", "extrema_score", "greedy_matching_score", "bleu", "rougeL", "meteor", "bertscore"]
        self.accumelated_instances = []
        
        #overlapping ngram metircs
        self.BLEU = evaluate.load('bleu')
        self.ROUGE = evaluate.load('rouge')
        self.METEOR = evaluate.load('meteor')
        self.BERTSCORE = evaluate.load("bertscore")
        self.CIDEr = Cider()

    def add_batch(self, preds, references):
        
        # compute embedding based metrics
        metrics = {
            "average_score" : AverageScore(self.tokenizer, self.embedding_layer),
            "extrema_score" : ExtremaScore(self.tokenizer, self.embedding_layer),
            "greedy_matching_score" : GreedyMatchingScore(self.tokenizer, self.embedding_layer)
        }

        result = {}
        # tokenize inputs, we have to ignore [SEP] , [START] tokens. 
        preds_tokenized = [self.tokenizer(pred, return_tensors="pt")['input_ids'].squeeze()[1:-1].cuda() for pred in preds]
        ref_tokenized = [self.tokenizer(ref, return_tensors="pt")['input_ids'].squeeze()[1:-1].cuda() for ref in references]

        for key in metrics:
            result[key] = metrics[key].compute(preds_tokenized, ref_tokenized)
        
        self.BLEU.add_batch(predictions=preds, references=references)
        self.ROUGE.add_batch(predictions=preds, references=references)
        self.METEOR.add_batch(predictions=preds, references=references)
        self.BERTSCORE.add_batch(predictions=preds, references=references)
        
        preds_tokenized = [p.cpu().tolist() for p in preds_tokenized]
        refs_tokenized = [r.cpu().tolist() for r in ref_tokenized]

        result[self.CIDEr.method()] = self.CIDEr.compute_score(preds_tokenized, refs_tokenized)

        self.accumelated_instances.append(result)
        return result

    def compute(self):
        result = {}
        result[self.BLEU.name] = self.BLEU.compute()
        result[self.ROUGE.name] = self.ROUGE.compute()
        result[self.METEOR.name] = self.METEOR.compute()
        result[self.BERTSCORE.name] = self.BERTSCORE.compute(lang='en')
        average_scores = []
        extrema_scores = []
        greedy_scores = []
        cider_scores = []
        #compute other metrics manually
        for item in self.accumelated_instances:
            average_scores.append(item['average_score'].mean)
            extrema_scores.append(item['extrema_score'].mean)
            greedy_scores.append(item['greedy_matching_score'].mean)
            cider_scores.append(item[self.CIDEr.method()][0])
        
        result['average_score'] = torch.mean(torch.stack(average_scores))
        result['extrema_score'] = torch.mean(torch.stack(extrema_scores))
        result['greedy_matching_score'] = torch.mean(torch.stack(greedy_scores))
        result[self.CIDEr.method()] = sum(cider_scores) / len(cider_scores)
        return result



if __name__ == '__main__':
    from transformers import LxmertTokenizer, LxmertModel

    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
    E = model.embeddings.word_embeddings.cuda()

    mc = MetricCalculator(tokenizer, E)
    preds = [['the cat is on the mat', 'the cat is on the mat'],  ['the cat is on the mat', 'the cat is on the mat']]
    target = [['there is a catty on the matew', 'a cat is on the mat'], ['there is a catty on the matew', 'a cat is on the mat']]
    for pred, ref in zip(preds, target):
        mc.add_batch(pred, ref)
    print(mc.compute())