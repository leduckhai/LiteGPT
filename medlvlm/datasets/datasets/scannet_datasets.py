import os
import json
import random
from plyfile import PlyData
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict


def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 7], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices


class ScannetVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, num_pọints = None):
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        if num_pọints is None:
            self.num_points = 200000
        else:
            self.num_points = num_pọints

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        with open(ann_path, 'r') as f:
            self.annotation = json.load(f)
        
        # data/Scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply
        exist_annotation = []
        self.point_paths = []
        for ann in self.annotation:
            fly_name = ann["scene_id"] + "_vh_clean_2.ply"
            point_path = os.path.join(self.vis_root, ann["scene_id"], fly_name)
            self.point_paths.append(point_path)
            if os.path.exists(point_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation


    def __len__(self):
        return len(self.annotation)

    def get_data(self, index):
        ann = self.annotation[index]

        fly_name = ann["scene_id"] + "_vh_clean_2.ply"
        point_path = os.path.join(self.vis_root, ann["scene_id"], fly_name)
        points_mesh = read_mesh_vertices_rgb(point_path)


        xyz = points_mesh[:, :3]
        color = points_mesh[:, 3: 6] / 255.0
        xyz_rgb = np.concatenate([xyz, color], axis=1)

        # Configure your number of samples, for faster loading
        self.rand_len = 4
        xyz_rgb_ = xyz_rgb
        for _ in range(self.rand_len):
            if len(xyz_rgb) > self.num_points:
                selected_indices = np.random.choice(len(xyz_rgb), self.num_points, replace=False)
                xyz_rgb_ = xyz_rgb[selected_indices]
            else:
                remaining_pts = self.num_points - len(xyz_rgb)
                selected_indices = np.random.choice(len(xyz_rgb), remaining_pts, replace=True)
                xyz_rgb_ = np.concatenate([xyz_rgb, xyz_rgb[selected_indices]], axis=0)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights


        return {
            "image": xyz_rgb_,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }
    


