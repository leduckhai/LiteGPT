from transformers.models.whisper.modeling_whisper import *
import torch.nn as nn

class WhisperForLiteGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.asr_encoder = WhisperForConditionalGeneration.from_pretrained("Hanhpt23/whisper-small-multimed").model.encoder
        self.d_model = self.asr_encoder.config.d_model
        self.pooling = nn.AdaptiveAvgPool2d((256, self.d_model)) # 1500 -> 256
        
    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = self.asr_encoder(
            input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        output = self.pooling(output)
        return output
    
def create_whisper(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = WhisperForLiteGPT()
    if precision == "fp16":
        model = model.half()
    return model