from .whisper import create_whisper

def build_audio_encoder(audio_model, **kwargs):
    if audio_model == "whisper":
        return create_whisper(**kwargs)