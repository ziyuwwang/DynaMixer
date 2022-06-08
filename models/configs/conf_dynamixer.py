from models.vision_model import VisionModel, _cfg
from timm.models.registry import register_model
from models.dynamixer import DynaMixerBlock

default_cfgs = {
    'DynaMixer_S': _cfg(crop_pct=0.9),
    'DynaMixer_M': _cfg(crop_pct=0.9),
    'DynaMixer_L': _cfg(crop_pct=0.875),
}


@register_model
def dynamixer_s(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_S']
    return model


@register_model
def dynamixer_m(pretrained=False, **kwargs):
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    resolutions = [32, 32, 16, 16]
    num_heads = [8, 8, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_M']
    return model


@register_model
def dynamixer_l(pretrained=False, **kwargs):
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_L']
    return model