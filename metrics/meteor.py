from evaluate import load

meteor_metric = load('meteor')

def compute_meteor(preds, labels):
    labels = [[label] for label in labels]
    result = meteor_metric.compute(predictions=preds, references=labels)
    result = {key: value * 100 for key, value in result.items()}
    return result