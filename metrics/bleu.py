from evaluate import load

bleu_metric = load("bleu")

def compute_bleu(preds, labels):
    labels = [[label] for label in labels]
    result = {f'bleu-{i}': bleu_metric.compute(predictions=preds, references=labels, max_order=i)['bleu'] for i in range(1,5)}
    result = {key: value * 100 for key, value in result.items()}
    return result