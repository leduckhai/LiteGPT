from open_clip import create_model_from_pretrained
import torch.nn as nn

class BiomedCLIP(nn.Module):
    def __init__(self):
        super(BiomedCLIP, self).__init__()
        self.model = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", return_transform=False).visual.trunk
        self.num_features = 768

    def forward(self, x):
        return self.model.forward_features(x)[:, 1:, :]
    
def create_biomed_clip(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = BiomedCLIP()
    if precision == "fp16":
        model = model.half()
    return model