import os
import json
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

class VinDrCXRDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, prompt_test=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        if prompt_test is None:
            self.instruction_pool = [
                "[grounding] please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes.",
                # "[grounding] please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes. Let's think step by step.",
                # "[grounding] This is a radiological image of a chest x-ray from a Posterior Anterior view. Examine all the areas where the lung borders the diaphragm, the heart and other mediastinal structures. Please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes.",
                # "[grounding] This is a radiological image of a chest x-ray from a Posterior Anterior view. Examine all the areas where the lung borders the diaphragm, the heart and other mediastinal structures. Please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes. Let's think step by step.",
                # "[grounding] Please describe this image in details with radiological features. Use one sentence unless there are no findings. The sentence should list the global diseases present in the image."
                # "[grounding] This is a radiological image of a chest x-ray from a Posterior Anterior view. Examine all the areas where the lung borders  the diaphragm, the heart and other mediastinal structures. Please describe this image in details with radiological features. Use one sentence unless there are no findings. The sentence should list the  global diseases present in the image."
            ]
        else:
            self.instruction_pool = [prompt_test]

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
        # instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            "audio": torch.rand(80, 3000, dtype=torch.float16),
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }
