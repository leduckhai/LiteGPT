from .eva_vit import create_eva_vit_g
from .clip_vit import create_pubmed_clip_vit
from .biomed_clip import create_biomed_clip
from .biomed_pubmed_clip import create_biomed_pubmed_clip
from .point_transformer import create_point_transformer

def build_vision_encoder(vision_model, **kwargs):
    if vision_model == "eva_clip_g":
        num_concat = 4
        return create_eva_vit_g(**kwargs), num_concat
    if vision_model == "pubmed_clip_vit":
        img_size = kwargs["img_size"]
        assert img_size == 224, "The resolution of the image must be (224, 224)"
        num_concat = 1
        return create_pubmed_clip_vit(**kwargs), num_concat
    if vision_model == "biomed_clip":
        img_size = kwargs["img_size"]
        assert img_size == 224, "The resolution of the image must be (224, 224)"
        num_concat = 4
        return create_biomed_clip(**kwargs), num_concat
    if vision_model == "biomed_pubmed_clip":
        img_size = kwargs["img_size"]
        assert img_size == 224, "The resolution of the image must be (224, 224)"
        num_concat = 5
        return create_biomed_pubmed_clip(**kwargs), num_concat
    if vision_model == "point_transformer":
        num_concat = 9
        return create_point_transformer(**kwargs), num_concat