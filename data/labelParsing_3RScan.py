import json
import csv
import re
import numpy as np
import pandas as pd# Use a pipeline as a high-level helper
from transformers import pipeline
import open3d as o3d
from collections import defaultdict


with open('data/3DSSG/objects.json', 'r') as f:
    objects_data = json.load(f)

with open('data/3DSSG/relationships.json', 'r') as f:
    relationships_data = json.load(f)

with open('data/3DSSG/classes.txt') as f:
    classes = f.readlines()
    id_to_text = {}
    for i in range(len(classes)):
        classes[i] = classes[i].split("\t")
        id_to_text[classes[i][0]] = " ".join(classes[i][1:])

id_to_class = defaultdict(str)
with open('data/3DSSG/wordnet_attributes.txt', 'r') as file:
    for idx, line in enumerate(file, start=1):
        class_name = line.strip().replace('\t', ' ')
        id_to_class[str(idx)] = class_name

def describe_object(obj):
    """Generate a text description of an object."""
    label = obj['label']
    label_id = obj["global_id"]
    if label_id not in id_to_text.keys():
        return ""
    description = f"{label + ' is a ' + id_to_text[label_id]}"
    if 'attributes' in obj:
        attr_descriptions = []
        for key, values in obj['attributes'].items():
            if key == "state":
                attr_descriptions.append(f"{'The state of ' + label + ' is ' + ', '.join(values)}")
            elif key == "color":
                attr_descriptions.append(f"{'The color of ' +label + ' is ' + ', '.join(values)}")
            elif key == "shape":
                attr_descriptions.append(f"{'The shape of ' +label + ' is ' + ', '.join(values)}")
            elif key == "material":
                attr_descriptions.append(f"{'The material of ' +label + ' is ' + ', '.join(values)}")
            elif key == "texture":
                attr_descriptions.append(f"{'The texture of ' +label + ' is ' + ', '.join(values)}")
            elif key == "symmetry":
                attr_descriptions.append(f"{'The symmetry of ' +label + ' is ' + ', '.join(values)}")
            elif key == "other":
                attr_descriptions.append(f"{'Other attributes of ' +label + ' are ' + ', '.join(values)}")
        description += " " + ", ".join(attr_descriptions)
    if 'affordances' in obj:
        description += ". It can be used for " + ", ".join(obj['affordances'])
    return description

def process_data(objects_data, relationships_data):
    json_data = []
    
    for scan in objects_data['scans']:
        scan_id = scan['scan']
        object_descriptions = []
        id_to_object = {}
        for obj in scan['objects']:
            if not any(o['id'] == obj['global_id'] for o in object_descriptions):
                desc = describe_object(obj)
                obj_entry = {
                    "label": obj["label"],
                    "id": str(obj['global_id']),
                    "desc": desc
                }
                
                id_to_object[obj['global_id']] = obj_entry
                object_descriptions.append(obj_entry)
        # print(id_to_object)
        for rel_scan in relationships_data['scans']:
            if rel_scan['scan'] == scan_id:
                # print(rel_scan['relationships'])
                for rel in rel_scan['relationships']:
                    from_id, to_id, _, relation = rel
                    if from_id > 528 or to_id > 528:
                        continue
                    from_id, to_id = str(from_id), str(to_id)
                    
        
                    from_obj = id_to_object.get(from_id, None)
                    to_obj = id_to_object.get(to_id, None)

                    if from_id in id_to_object:
                        from_obj = id_to_object[from_id]
                    else:
                        from_obj = {
                            "label": id_to_class[from_id],
                            "id": from_id,
                            "desc": id_to_text[from_id]
                        }
                    if to_id in id_to_object:
                        to_obj = id_to_object[to_id]
                    else:
                        to_obj = {
                            "label": id_to_class[to_id],
                            "id": to_id,
                            "desc": id_to_text[to_id]
                        }

                    relation_desc = f"{from_obj['label']} is {relation} {to_obj['label']}"
     
                    from_obj["desc"]+= ". " + relation_desc
                    contain_from = False
                    for i,o in enumerate(object_descriptions):
                        if o['id'] == from_id:
                            contain_from = True
                            object_descriptions[i]['desc'] = from_obj['desc']
                    if not contain_from:
                        object_descriptions.append(from_obj)
                        id_to_object[from_id] = from_obj


        scan_data = {
            "scan_id": scan_id,
            "objects": object_descriptions
        }
        if scan_id == "baf0a8fb-26d4-2033-8a28-2001356bbb9a":
            json_data.append(scan_data)

    with open("text_label.json", "w") as outfile: 
        json.dump(json_data, outfile)


