"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import contextlib

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from medlvlm.common.dist_utils import download_cached_file
from medlvlm.common.utils import get_abs_path, is_url
from .vision_model.builder import build_vision_encoder

from transformers import AutoTokenizer



class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[-1].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_vision_encoder(
        cls, model_name, freeze, **kwargs
    ):
        logging.info(f'Loading {model_name}')

        precision = kwargs.get('precision', "fp16")
        if not freeze:
            if precision is not None:
                kwargs["precision"] = "fp32"  # fp16 is not for training

        visual_encoder, num_concat = build_vision_encoder(model_name, **kwargs)

        ln_vision = LayerNorm(visual_encoder.num_features)

        if freeze:
            for param in visual_encoder.parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for param in ln_vision.parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        logging.info(f'Loading {model_name} Done')
        return visual_encoder, ln_vision, num_concat

    def init_llm(cls, language_model_path, bits=8, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        logging.info(f'Loading language model at {language_model_path}')

  
        tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=False)
        tokenizer.pad_token = "$$"

        model_args = {}
        if low_resource:
            from transformers import BitsAndBytesConfig
            model_args.update(dict(
                pretrained_model_name_or_path=language_model_path,
                device_map={"": low_res_device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit= bits == 4,
                    load_in_8bit= bits == 8,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            ))
        else:
            model_args = {
                "pretrained_model_name_or_path": language_model_path,
                "torch_dtype": torch.float16
            }

        if "llama" in language_model_path.lower():
            from medlvlm.models.language_model.modeling_llama import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(**model_args)
        elif "mistral" in language_model_path.lower():
            from medlvlm.models.language_model.modeling_mistral import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(**model_args)
        else:
            pass

        if lora_r > 0:
            model = prepare_model_for_kbit_training(model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            model = get_peft_model(model, loraconfig)
            for name, module in model.named_modules():
                if 'norm' in name:
                    module = module.to(torch.float16)

                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        module = module.to(torch.float16)

            model.print_trainable_parameters()

        else:
            for param in model.parameters():
                param.requires_grad = False
        logging.info(f'Loading language model Done!')
        return model, tokenizer


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)