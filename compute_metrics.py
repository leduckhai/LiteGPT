import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from metrics import *
from medlvlm.common.eval_utils import computeIoU
from tqdm import tqdm
from collections import defaultdict

CLASS_NAMES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Other lesion", "COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other disease", "No finding", "Finding"]

def compute_text_validity(preds, labels):
    results = compute_rouge(preds, labels)
    results |= compute_bleu(preds, labels)
    results |= compute_meteor(preds, labels)
    results |= {"cider": compute_cider(preds, labels)}
    return {k: round(v, 4) for k, v in results.items()}

def remove_bboxes(text):
    cleaned_text = re.sub(r"{<\d+><\d+><\d+><\d+>}", "", text)
    return cleaned_text

def extract_labels(text):
    labels = [0] * len(CLASS_NAMES)
    text = re.sub(r"<.*?>", "", text.lower())
    for i, label in enumerate(CLASS_NAMES[:-1]):
        if label.lower() in text:
            labels[i] = 1
    if not labels[-2]:
        labels[-1] = 1
    return labels

def extract_labels_global(text):
    labels = [0] * len(CLASS_NAMES[22:])
    text = re.sub(r"<.*?>", "", text.lower())
    for i, label in enumerate(CLASS_NAMES[22:-1]):
        if label.lower() in text:
            labels[i] = 1
    if not labels[-2]:
        labels[-1] = 1
    return labels

def extract_labels_local(text):
    labels = [0] * len(CLASS_NAMES[:22])
    text = re.sub(r"<.*?>", "", text.lower())
    for i, label in enumerate(CLASS_NAMES[:22]):
        if label.lower() in text:
            labels[i] = 1
    return labels

def show_numerical_report(pred_path):
    with open(pred_path, "r") as f:
        json_data = json.load(f)
    preds = [remove_bboxes(data["predict"]) for data in json_data]
    labels = [remove_bboxes(data["ground_truth"]) for data in json_data]
    return compute_metrics(preds, labels)

def show_classification_report(gt_path, pred_local_path=None, pred_global_path=None, combine_path=None):
    """
        gt_path: path to csv file (e.g image_label_text.csv)
        pred_local_path: path to json file
        pred_global_path: path to json file
    """
    gt = pd.read_csv(gt_path)
    if pred_local_path is not None:
        gt_vec = gt.iloc[:, 1:23].values

        data_pred_local = json.load(open(pred_local_path, 'r'))
        preds = {data["image_id"]: extract_labels_local(data["predict"]) for data in data_pred_local}

        pred_vec = np.vstack([preds[image_id] for image_id in gt["image_id"] if preds.get(image_id, None) is not None])
        gt_vec = np.stack([gt_vec[i] for i, image_id in enumerate(gt["image_id"]) if preds.get(image_id, None) is not None])
        print(classification_report(gt_vec, pred_vec, target_names=CLASS_NAMES[:22]))

    if pred_global_path is not None:
        gt["Finding"] = 1 - gt["No finding"]
        gt_vec = gt.iloc[:, 23:].values

        data_pred_global = json.load(open(pred_global_path, 'r'))
        preds = {data["image_id"]: extract_labels_global(data["predict"]) for data in data_pred_global}

        pred_vec = np.vstack([preds[image_id] for image_id in gt["image_id"] if preds.get(image_id, None) is not None])
        gt_vec = np.stack([gt_vec[i] for i, image_id in enumerate(gt["image_id"]) if preds.get(image_id, None) is not None])
        print(classification_report(gt_vec, pred_vec, target_names=CLASS_NAMES[22:]))

    if combine_path is not None:
        gt["Finding"] = 1 - gt["No finding"]
        gt_vec = gt.iloc[:, 1:].values

        data_pred_combine = json.load(open(combine_path, 'r'))
        preds = {data["image_id"]: extract_labels(data["predict"]) for data in data_pred_combine}

        pred_vec = np.vstack([preds[image_id] for image_id in gt["image_id"] if preds.get(image_id, None) is not None])
        gt_vec = np.stack([gt_vec[i] for i, image_id in enumerate(gt["image_id"]) if preds.get(image_id, None) is not None])
        print(classification_report(gt_vec, pred_vec, target_names=CLASS_NAMES))

def compute_accuracy(pred_path, threshold=0.5):
    with open(pred_path, "r") as f:
        predict_data = json.loads(f.read())
    
    results_pred = {}
    for data in predict_data:
        predict = data["predict"]
        image_id = data["image_id"]
        pattern = r'<p>([^<]+)</p> \{<(\d+)><(\d+)><(\d+)><(\d+)>\}'
        matches = re.findall(pattern, predict)
        if len(matches) == 0:
            continue
        results_pred[image_id] = [{'local_name': name, 'bounding_box': (int(x1), int(y1), int(x2), int(y2))} for name, x1, y1, x2, y2 in matches]
    
    results_gt = {}
    for data in predict_data:
        gt = data["ground_truth"]
        image_id = data["image_id"]
        pattern = r'<p>([^<]+)</p> \{<(\d+)><(\d+)><(\d+)><(\d+)>\}'
        matches = re.findall(pattern, gt)
        if len(matches) == 0:
            continue
        results_gt[image_id] = [{'local_name': name, 'bounding_box': (int(x1), int(y1), int(x2), int(y2))} for name, x1, y1, x2, y2 in matches]
    
    dict_pred = []
    for image_id, d in tqdm(results_pred.items(), total=len(results_pred)):
        for x in d:
            dict_pred.append({
                "image_id": image_id,
                "class_name": x["local_name"],
                "x_min": x["bounding_box"][0],
                "y_min": x["bounding_box"][1],
                "x_max": x["bounding_box"][2],
                "y_max": x["bounding_box"][3]
            })
            
    dict_gt = []
    for image_id, d in tqdm(results_gt.items(), total=len(results_gt)):
        for x in d:
            dict_gt.append({
                "image_id": image_id,
                "class_name": x["local_name"],
                "x_min": x["bounding_box"][0],
                "y_min": x["bounding_box"][1],
                "x_max": x["bounding_box"][2],
                "y_max": x["bounding_box"][3]
            })

    pred_df = pd.DataFrame(dict_pred)
    gt_df = pd.DataFrame(dict_gt)

    total = len(gt_df)
    
    gt_dict = defaultdict(dict)
    for gt in gt_df.iloc:
        class_name = gt["class_name"]
        image_id = gt["image_id"]
        
        if gt_dict[image_id].get(class_name, None) is not None:
            gt_dict[image_id][class_name].append([gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]])
        else:
            gt_dict[image_id][class_name] = [[gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]]]
            
    pred_dict = defaultdict(dict)
    for pred in pred_df.iloc:
        class_name = pred["class_name"]
        image_id = pred["image_id"]
        
        if pred_dict[image_id].get(class_name, None) is not None:
            pred_dict[image_id][class_name].append([pred["x_min"], pred["y_min"], pred["x_max"], pred["y_max"]])
        else:
            pred_dict[image_id][class_name] = [[pred["x_min"], pred["y_min"], pred["x_max"], pred["y_max"]]]

    count = 0
    for image_id in gt_dict.keys():
        for class_name, gt_bboxes in gt_dict[image_id].items():
            try:
                pred_bboxes = pred_dict[image_id][class_name]
                for gt_bbox in gt_bboxes:
                    for pred_bbox in pred_bboxes:
                        iou = computeIoU(pred_bbox, gt_bbox)
                        if iou > threshold:
                            count += 1
                            break
            except:
                pass

    return count / total * 100