import pandas as pd
import os
import glob
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import json
from medlvlm.common.eval_utils import computeIoU
import itertools
from collections import defaultdict

IMAGE_FOLDER = "path/to/image_folder"

def groupby_bboxes_overlap(df):
    df["overlap_group"] = -1
    
    for name, group in df.groupby(["image_id", "class_name"]):
        if name[1] == "No finding":
            continue

        bboxes = group[['x_min', 'y_min', 'x_max', 'y_max']].values
        group_assignments = [0] * bboxes.shape[0]
        next_group_id = 1

        for idx_bbox1, idx_bbox2, in itertools.combinations(range(bboxes.shape[0]), 2):
            bbox1, bbox2 = bboxes[idx_bbox1], bboxes[idx_bbox2]
            iou = computeIoU(bbox1, bbox2)
            if iou > 0.5:
                if group_assignments[idx_bbox1] == 0 and group_assignments[idx_bbox2] == 0:
                    group_assignments[idx_bbox1] = group_assignments[idx_bbox2] = next_group_id
                    next_group_id += 1
                elif group_assignments[idx_bbox1] != 0:
                    group_assignments[idx_bbox2] = group_assignments[idx_bbox1]
                else:
                    group_assignments[idx_bbox1] = group_assignments[idx_bbox2]
        
        next_group_id = max(group_assignments) + 1
        for i in range(len(group_assignments)):
            if group_assignments[i] == 0:
                group_assignments[i] = next_group_id
                next_group_id += 1

        df.loc[group.index, "overlap_group"] = group_assignments
    return df

def get_image_size(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    image_id = os.path.basename(path).split(".")[0]
    return image_id, width, height

def preprocess_local_labels(path, process_overlap=True):
    df = pd.read_csv(path)

    if process_overlap:
        df = groupby_bboxes_overlap(df)
        df.drop_duplicates(subset=['image_id', 'class_name', 'overlap_group'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

    images_size = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_image_size, path) for path in paths]
        
        for future in tqdm(as_completed(futures), total=len(paths)):
            image_id, width, height = future.result()
            images_size[image_id] = [width, height]

    sizes = np.array([images_size[image_id] for image_id in df["image_id"]])
    sizes = np.tile(sizes, 2)
    df.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']] = round(df.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']] / sizes * 100)

    df["labels"] = df.apply(lambda x: x["class_name"] if x["class_name"] == "No finding" else f"<p>{x['class_name']}</p> {{<{int(x['x_min'])}><{int(x['y_min'])}><{int(x['x_max'])}><{int(x['y_max'])}>}}", axis=1)
    df_labels = df.groupby('image_id')['labels'].apply(lambda x: ','.join(set(x))).reset_index()
    local_labels = [{"image_id": image_id, "local_labels": f"Local diseases of this chest radiograph are {label}."} for image_id, label in zip(df_labels['image_id'], df_labels['labels'])]
    return local_labels

def preprocess_global_labels(path):
    df = pd.read_csv(path)
    df.rename(columns = {'Other disease':'Other diseases'}, inplace = True)

    global_disease_names = ["Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", "COPD", "No finding"]
    results = defaultdict(set)
    for row in tqdm(df.iloc, total=len(df)):
        indexes = {i for i, global_disease_name in enumerate(global_disease_names) if row[global_disease_name]}
        results[row["image_id"]].update(indexes)
    results = {image_id: [global_disease_names[index] for index in indexes] for image_id, indexes in results.items()}
    global_labels = [{"image_id": image_id, "global_labels": "Global diseases of this chest radiograph are {}.".format(", ".join(global_disease_names))} for image_id, global_disease_names in results.items()]
    return global_labels

def main():
    ann_path = "path/to/annotations_train.csv"
    image_labels_path = "path/to/image_labels_train.csv"

    local_labels = preprocess_local_labels(ann_path, process_overlap=True)
    global_labels = preprocess_global_labels(image_labels_path)

    combine_labels = []
    id2item_local = {local_label["image_id"]: local_label for local_label in local_labels}
    for item in global_labels:
        image_id, label = item["image_id"], item["global_labels"]
        if "No finding" in label:
            combine_labels.append({
                "image_id": image_id,
                "grounded_diseases": label
            })
        else:
            local_label = id2item_local[image_id]["local_labels"]
            combine_labels.append({
                "image_id": image_id,
                "grounded_diseases": f"{label} {local_label}"
            })

    with open('local_labels_train_v3.json', 'w', encoding='utf-8') as f:
        json.dump(local_labels, f, ensure_ascii=False, indent=4)

    with open('global_labels_train_v3.json', 'w', encoding='utf-8') as f:
        json.dump(global_labels, f, ensure_ascii=False, indent=4)

    with open('grounded_diseases_train_v3.json', 'w', encoding='utf-8') as f:
        json.dump(combine_labels, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()