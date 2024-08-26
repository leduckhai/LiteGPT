import json
import csv
import re
import numpy as np
import pandas as pd# Use a pipeline as a high-level helper
from transformers import pipeline
from scannet_utils import *
import open3d as o3d
from collections import defaultdict

def getQAByScene(qaPath):
    with open(qaPath, 'r') as f:
        qa = json.load(f)

    scene_dict = defaultdict(lambda: {"qa": []})
    id_to_category, category_to_id, _, _,raw_to_category = getLabelMapping()
    for entry in qa:
        scene_id = entry["scene_id"]
        ids = entry["object_ids"]
        obj_names = entry["object_names"]

        for i in range(len(ids)):
            obj_names[i] = raw_to_category[obj_names[i]]
            ids[i] = category_to_id[obj_names[i]]

        qa_entry = {
            "question_id": entry["question_id"],
            "question": entry["question"],
            "answers": entry["answers"],
            "object_ids": ids,
            "object_names": obj_names,
        }
        scene_dict[scene_id]["qa"].append(qa_entry)

    # Convert the defaultdict back to a regular dictionary if needed
    scene_dict = dict(scene_dict)
    with open('data/qa/scene_to_QA.json', 'w') as f:
        json.dump(scene_dict, f, indent=4)
    return scene_dict

def getLabelMapping():
    path = "data/qa/scannetv2-labels.combined.tsv"
    df = pd.read_csv(path, sep='\t')
    id_to_category = {}
    category_to_id= {}
    raw_to_category = {}
    nyuid_to_nyuclass = defaultdict(str)
    nyuclass_to_nyuid = defaultdict(int)
    for i, row in df.iterrows():
        category_to_id[row['category']] = row['id']
        id_to_category[row['id']] = row['category']
        raw_to_category[row['raw_category']] = row['category']
        nyuid_to_nyuclass[row['nyu40id']] = row['nyu40class']
        nyuclass_to_nyuid[row['nyu40class']] = row['nyu40id']

    return id_to_category, category_to_id, nyuid_to_nyuclass, nyuclass_to_nyuid, raw_to_category
    

def visualize_scene(path):
    mesh = o3d.io.read_triangle_mesh(path)
    o3d.visualization.draw_geometries([mesh])
    return mesh


# qaPath = "data/qa/ScanQA_v1.0_train.json"
# scene_dict = getQAByScene(qaPath)
