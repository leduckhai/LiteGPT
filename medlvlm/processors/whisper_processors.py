"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from medlvlm.common.registry import registry
from omegaconf import OmegaConf
from medlvlm.processors.base_processor import BaseProcessor
from transformers import WhisperProcessor


@registry.register_processor("whisper_processor")
class WhisperAudioProcessor(BaseProcessor):
    def __init__(self, model_name="openai/whisper-tiny", sampling_rate=16000):
        self.audio_processor = WhisperProcessor.from_pretrained(model_name)
        self.sampling_rate = sampling_rate

    def __call__(self, waveform):
        # Process the waveform using the WhisperProcessor
        return self.audio_processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt").input_features

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        model_name = cfg.get("model_name", "openai/whisper-tiny")
        sampling_rate = cfg.get("sampling_rate", 16000)

        return cls(model_name=model_name, sampling_rate=sampling_rate)
