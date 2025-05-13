import torch
import torch.nn.functional as F
from PIL import Image

from custom_siglip import preprocess_image, SigLipVisionModel, SigLipVisionConfig 

from transformers import SiglipVisionModel as OriginalSiglipVisionModel
from transformers import SiglipVisionConfig as OriginalSiglipVisionConfig

def load_pretrained_weights(custom_model, pretrained_model):
    # Load the pretrained weights from the original model
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = custom_model.state_dict()

    mapped_dict = {}

    for key, value in pretrained_dict.items():
        # vision_model -> transformer
        if key.startswith('vision_model.'):
            new_key = key.replace('vision_model.', 'transformer.')
        else:
            new_key = key
        
        # layers -> encoder_blocks
        if 'encoder.layers.' in new_key:
            new_key = new_key.replace('encoder.layers.', 'encoder.encoder_blocks.')
        
        # 커스텀 모델에 해당 키가 있으면 매핑
        if new_key in custom_dict:
            mapped_dict[new_key] = value

    custom_model.load_state_dict(mapped_dict, strict=False)

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

    print("siglip_custom.shape: ", custom_out.shape)
    print("siglip_original.shape: ", original_out[0].shape)
    print("siglip_original-afterPooling.shape: ", original_out[1].shape)
    cosine_similarity = F.cosine_similarity(custom_out, original_out[0])
    print("cosine_similarity.mean(): ", cosine_similarity.mean())
    print("cosine_similarity.min(): ", cosine_similarity.min())