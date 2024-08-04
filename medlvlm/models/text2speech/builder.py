from .whisper import create_whisper

def build_audio_encoder(vision_model, **kwargs):
    if vision_model == "whisper":
        num_concat = 4
        return create_whisper(**kwargs), num_concat