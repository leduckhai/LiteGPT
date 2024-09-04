import os
import json
import random
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class AudioInstruction(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, audio_dir, vis_root, ann_path, prompt_test=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.audio_dir = audio_dir

        self.vis_processor = vis_processor

        self.text_processor = text_processor
        self.audio_processor = audio_processor

        # Load the audio file paths
        self.audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
        
        # Preload all audio files
        self.audio_data = []
        for audio_file in self.audio_files:
            audio_path = os.path.join(self.audio_dir, audio_file)
            waveform, sample_rate = torchaudio.load(audio_path)
            self.audio_data.append((waveform, sample_rate, audio_file))

        if prompt_test is None:
            self.instruction_pool = [
                "[grounding] please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes."
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

        answer = info['global_labels']
        # instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img>" 

        # audio
        waveform, sample_rate, audio_file = random.choice(self.audio_data) # sample_rate = 24000

        # print('shape of audio before:', waveform.shape, sample_rate)

        waveform_array = waveform.squeeze().numpy()

        waveform = self.audio_processor(waveform_array) #, sampling_rate=16000, return_tensors="pt").input_features
        waveform = waveform.squeeze()

        return {
            "image": image,
            "audio": waveform, #torch.rand(80, 3000, dtype=torch.float16), # double length of the max_source_positions
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_id'],
        }
