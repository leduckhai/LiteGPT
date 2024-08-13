import os
import json
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

class VinDrCXRDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        # image_file = 'COCO_train2014_{}.jpg'.format(info['image_id'])
        image_file = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['grounded_diseases']
        instruction = "<Img><ImageHere></Img>"
        audio = torch.rand(80, 3000, dtype=torch.float16)

        return {
            "image": image,
            "audio": audio,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }
