import torch
import evaluate
from src.metrics.EmbeddingBase.AverageScore import AverageScore
from src.metrics.EmbeddingBase.ExtremaScore import ExtremaScore
from src.metrics.EmbeddingBase.GreedyMatchingScore import GreedyMatchingScore

#!pip install evaluate
#!pip install rouge_score
#!pip install bert_score

class MetricCalculator():
    def __init__(self, tokenizer, embedding_layer) -> None:
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
    def compute(self, preds, references):
        
        # compute embedding based metrics
        metrics = {
            "average_score" : AverageScore(self.tokenizer, self.embedding_layer),
            "extrema_score" : ExtremaScore(self.tokenizer, self.embedding_layer),
            "greedy_matching_score" : GreedyMatchingScore(self.tokenizer, self.embedding_layer)
        }

        result = {}
        preds_tokenized = [self.tokenizer(pred, return_tensors="pt")['input_ids'].squeeze() for pred in preds]
        ref_tokenized = [self.tokenizer(ref, return_tensors="pt")['input_ids'].squeeze() for ref in references]

        for key in metrics:
            result[key] = metrics[key].compute(preds_tokenized, ref_tokenized)
        

        #compute overlapping ngrams metrics
        metrics = {
            #bleu.compute(predictions=preds, references=target)
            'bleu' : evaluate.load('bleu'),
            #rouge.compute(predictions=preds, references=target)
            'rougeL' : evaluate.load('rouge'),
            #meteor.compute(predictions=preds, references=target)
            'meteor' : evaluate.load('meteor'),            
        }
        
        for key in metrics:
            result[key] = metrics[key].compute(predictions=preds, references=references)
        
        bert_score = evaluate.load("bertscore")
        result['bertscore'] =  bert_score.compute(predictions = preds, references = references, lang='en')


        return result



if __name__ == '__main__':
    from transformers import LxmertTokenizer, LxmertModel

    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
    E = model.embeddings.word_embeddings

    mc = MetricCalculator(tokenizer, E)
    preds = ['the cat is on the mat', 'the cat is on the mat']
    target = ['there is a catty on the matew', 'a cat is on the mat']
    mc.compute(preds, target)