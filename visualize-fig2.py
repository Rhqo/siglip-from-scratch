import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

from custom_siglip import preprocess_image, SigLipVisionConfig, SigLipVisionModel

from transformers import SiglipVisionModel as OriginalSiglipVisionModel
from transformers import SiglipVisionConfig as OriginalSiglipVisionConfig

from compare_origin import load_pretrained_weights

image = Image.open("image.jpg")
image_tensor = preprocess_image(image)

config = SigLipVisionConfig()
custom_model = SigLipVisionModel(config)

original_config = OriginalSiglipVisionConfig(vision_use_head=True)
original_model = OriginalSiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", config=original_config)

load_pretrained_weights(custom_model, original_model)

with torch.no_grad():
    # (1, 3, 224, 224) -> [(1, 196, 768), (1, 768)]
    last_hidden_state = custom_model(image_tensor)


last_hidden_state_np = last_hidden_state[0].squeeze(0).cpu().numpy()
output_vector_np = last_hidden_state[1].squeeze(0).cpu().numpy()

# Reshape to (num_patches, embed_dim)
last_hidden_state_np = last_hidden_state_np.reshape(-1, config.embed_dim)

# Normalize the values to [0, 1] for visualization
last_hidden_state_np = (last_hidden_state_np - np.min(last_hidden_state_np)) / (np.max(last_hidden_state_np) - np.min(last_hidden_state_np))

# Reshape to (num_patches, num_patches, embed_dim)
num_patches = int(np.sqrt(last_hidden_state_np.shape[0]))
last_hidden_state_np = last_hidden_state_np.reshape(num_patches, num_patches, config.embed_dim)

last_hidden_state_np = last_hidden_state_np.mean(-1)

plt.figure(figsize=(10, 5))

# 1. Last hidden state
plt.subplot(1, 2, 1)
plt.imshow(last_hidden_state_np, cmap='viridis')
plt.colorbar()
plt.title("Last Hidden State")
plt.axis('off')

# 2. Output vector
plt.subplot(1, 2, 2)
plt.title('Output vector')
plt.plot(output_vector_np)
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.tight_layout()

plt.show()