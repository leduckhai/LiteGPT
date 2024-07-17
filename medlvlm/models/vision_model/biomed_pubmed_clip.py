import torch.nn as nn
import torch
from open_clip import create_model_from_pretrained
from transformers import CLIPModel

class BiomedPubmedCLIP(nn.Module):
    def __init__(self):
        super(BiomedPubmedCLIP, self).__init__()
        self.biomed_clip = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", return_transform=False).visual.trunk
        self.pubmed_clip = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").vision_model
        self.num_features = 768
    def forward(self, x):
        x1 = self.biomed_clip.forward_features(x)[:, 1:, :]
        x2 = self.pubmed_clip(x).last_hidden_state[:, 1:, :]
        x = torch.cat([x1, x2], dim=1)
        return x
    
def create_biomed_pubmed_clip(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = BiomedPubmedCLIP()
    if precision == "fp16":
        model = model.half()
    return model