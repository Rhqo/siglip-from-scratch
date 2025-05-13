import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from custom_siglip import preprocess_image, VisionEmbeddings, SigLipVisionConfig

image = Image.open('image.jpg').convert('RGB')
config = SigLipVisionConfig()
image_tensor = preprocess_image(image, image_size=config.image_size)


vision_embeddings = VisionEmbeddings(config)
with torch.no_grad():
    patch_embeds = vision_embeddings.patch_embedding(image_tensor)  # (B, embed_dim, 14, 14)
    patch_embeds_2d = patch_embeds[0].permute(1, 2, 0).reshape(config.image_size // config.patch_size,
                                                              config.image_size // config.patch_size,
                                                              config.embed_dim)  # (14, 14, 768)
    patch_embeds_mean = patch_embeds_2d.mean(-1).cpu().numpy()  # (14, 14)

    # Positional embedding
    pos_embeds = vision_embeddings.position_embedding.weight  # (196, 768)
    pos_embeds_2d = pos_embeds.reshape(config.image_size // config.patch_size,
                                       config.image_size // config.patch_size,
                                       config.embed_dim)  # (14, 14, 768)
    pos_embeds_mean = pos_embeds_2d.mean(-1).cpu().numpy()  # (14, 14)

plt.figure(figsize=(12, 5))

# Patch embedding heatmap
plt.subplot(1, 2, 1)
plt.title('Patch Embedding (mean over dim)')
plt.imshow(patch_embeds_mean, cmap='viridis')
plt.xlabel('Patch X')
plt.ylabel('Patch Y')
plt.colorbar()

# Positional embedding heatmap
plt.subplot(1, 2, 2)
plt.title('Positional Embedding (mean over dim)')
plt.imshow(pos_embeds_mean, cmap='plasma')
plt.xlabel('Patch X')
plt.ylabel('Patch Y')
plt.colorbar()

plt.tight_layout()
plt.show()
