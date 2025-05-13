import torch
import torch.nn.functional as F
from PIL import Image

from custom_siglip import preprocess_image, SigLipVisionModel, SigLipVisionConfig 

from transformers import SiglipVisionModel as OriginalSiglipVisionModel
from transformers import SiglipVisionConfig as OriginalSiglipVisionConfig

def load_pretrained_weights(custom_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = custom_model.state_dict()
    
    mapped_dict = {}
    
    # original: custom
    key_replacements = {
        'vision_model.': 'transformer.',
        'encoder.layers.': 'encoder.encoder_blocks.',
        'head.': 'attn_pooling_head.'
    }
    
    total_keys = len(pretrained_dict)
    mapped_count = 0
    shape_mismatch_count = 0
    
    for key, value in pretrained_dict.items():
        new_key = key
        for old_str, new_str in key_replacements.items():
            if old_str in new_key:
                new_key = new_key.replace(old_str, new_str)
        
        if new_key in custom_dict:
            if value.shape == custom_dict[new_key].shape:
                mapped_dict[new_key] = value
                mapped_count += 1
            else:
                shape_mismatch_count += 1
    
    # Load weights into custom model
    custom_model.load_state_dict(mapped_dict, strict=False)
    
    print(f"Mapped {mapped_count} out of {total_keys} keys")


if __name__ == "__main__":
    image = Image.open("image.jpg")
    # (3, 224, 224) -> (1, 3, 224, 224)
    image_tensor = preprocess_image(image)

    config = SigLipVisionConfig(
        num_channels=3,
        embed_dim=768,
        image_size=224,
        patch_size=16,
        num_attention_heads=12,
        attention_dropout=0.0,
        num_encoder_blocks=12,
        mlp_hidden_dim=3072,
        layer_norm_eps=1e-6
    )

    custom_model = SigLipVisionModel(config)

    original_config = OriginalSiglipVisionConfig(vision_use_head=True)
    original_model = OriginalSiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", config=original_config)

    # print("custom_model : ", custom_model)
    # print("original_model : ", original_model)

    load_pretrained_weights(custom_model, original_model)

    with torch.no_grad():
        # (1, 3, 224, 224) -> (1, 196, 768)
        custom_out = custom_model(image_tensor)
        original_out = original_model(image_tensor)

    print("siglip_custom-last_hidden_state.shape: ", custom_out[0].shape)
    print("siglip_original-last_hidden_state.shape: ", original_out[0].shape)
    print("siglip_original-afterPooling.shape: ", original_out[1].shape)
    cosine_similarity = F.cosine_similarity(custom_out[1], original_out[1])
    print("cosine_similarity.mean(): ", cosine_similarity.mean())
    print("cosine_similarity.min(): ", cosine_similarity.min())