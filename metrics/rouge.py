from evaluate import load
import nltk

rouge_metric = load("rouge")

def compute_rouge(preds, labels):
    # Rouge expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = rouge_metric.compute(predictions=preds, references=labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    return result