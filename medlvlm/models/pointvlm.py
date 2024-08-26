import logging
import random
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from medlvlm.common.registry import registry
from medlvlm.models.base_model import disabled_train
from medlvlm.models.pointvlm_base import PointVLMBase

IMG_DIM_VIT_LLAMA = 5632 # 1408 * 4

@registry.register_model("pointvlm")
class PointVLM(PointVLMBase):
    """
    MedLVLM model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/medlvlm.yaml",
    }

    def __init__(
            self,
            vision_model="point_transformer",
            audio_model="whisper",
            drop_path_rate=0,
            use_grad_checkpoint=False,
            precision="fp16",
            freeze_vision=True,
            freeze_audio=True,
            language_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            bits=8,
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vision_model=vision_model,
            audio_model=audio_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            precision=precision,
            freeze_vision=freeze_vision,
            freeze_audio=freeze_audio,
            language_model=language_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            bits=bits,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        img_f_dim = self.visual_encoder.num_features * self.num_concat
        if vision_model == "point_transformer" and "llama" in language_model: 
            self.language_proj = nn.Linear(
                img_f_dim, self.language_model.config.hidden_size
            )
        else:
            self.language_proj = nn.Sequential(
                nn.Linear(img_f_dim, IMG_DIM_VIT_LLAMA),
                nn.GELU(),
                nn.Linear(IMG_DIM_VIT_LLAMA, self.language_model.config.hidden_size)
            )

        self.audio_language_proj = nn.Linear(self.audio_encoder.d_model, self.language_model.config.hidden_size)

        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.language_model.gradient_checkpointing_enable()

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            x = self.visual_encoder(image)
            image_embeds = self.ln_vision(x).to(device)
            # image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            # [bs, 513, 384]
            image_embeds = image_embeds.view(bs, int(pn / self.num_concat), int(hs * self.num_concat))

            inputs_language = self.language_proj(image_embeds)
            atts_language = torch.ones(inputs_language.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_language, atts_language
    
    def encode_audio(self, audio):
        device = audio.device

        with self.maybe_autocast():
            audio_embeds = self.audio_encoder(audio).to(device)

            inputs_language = self.audio_language_proj(audio_embeds)
            atts_language = torch.ones(inputs_language.size()[:-1], dtype=torch.long).to(audio.device)
        return inputs_language, atts_language

    @classmethod
    def from_config(cls, cfg):
        vision_model = cfg.get("vision_model", "point_transformer")
        audio_model = cfg.get("audio_model", "whisper")
        language_model = cfg.get("language_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        precision = cfg.get("precision", "fp16")
        freeze_vision = cfg.get("freeze_vision", True)
        freeze_audio = cfg.get("freeze_audio", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        bits = cfg.get("bits", 8)
        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vision_model=vision_model,
            audio_model=audio_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            precision=precision,
            freeze_vision=freeze_vision,
            freeze_audio=freeze_audio,
            language_model=language_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            bits=bits,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Model Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

            if os.path.basename(ckpt_path) == "checkpoint_stage3.pth" and "llama" in language_model:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        name = name.replace("language_model", "llama_model")
                        name = name.replace("default.", "")
                        if ckpt['model'].get(name, None) is not None:
                            param.copy_(ckpt['model'][name])
                            
                    if vision_model == "eva_clip_g":
                        model.language_proj.weight.copy_(ckpt['model']['llama_proj.weight'])
                        model.language_proj.bias.copy_(ckpt['model']['llama_proj.bias'])
                    else:
                        model.language_proj[-1].weight.copy_(ckpt['model']['llama_proj.weight'])
                        model.language_proj[-1].bias.copy_(ckpt['model']['llama_proj.bias'])

        return model