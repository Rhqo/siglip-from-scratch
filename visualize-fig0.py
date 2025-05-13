import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from custom_siglip import preprocess_image, VisionEmbeddings, SigLipVisionConfig

image = Image.open("image.jpg")
image_tensor = preprocess_image(image)

config = SigLipVisionConfig()
vision_embeddings = VisionEmbeddings(config)

embeddings = vision_embeddings(image_tensor)

plt.figure(figsize=(12, 5))

# 1. Original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

# 2. Preprocessed image
img_np = image_tensor[0].permute(1, 2, 0).detach().numpy()
# Denormalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np = std * img_np + mean
img_np = np.clip(img_np, 0, 1)

plt.subplot(1, 3, 2)
plt.title('Preprocessed Image\n(After normalization)')
plt.imshow(img_np)
plt.axis('off')

# 3. Show image with patch grid
plt.subplot(1, 3, 3)
plt.title('Image with Patch Grid (16x16 pixels)')
grid_img = img_np.copy()
h, w = grid_img.shape[0], grid_img.shape[1]
patch_size = 16

for i in range(0, h, patch_size):
    grid_img[i:i+1, :] = [1, 1, 1]

for i in range(0, w, patch_size):
    grid_img[:, i:i+1] = [1, 1, 1]

plt.imshow(grid_img)
plt.axis('off')

plt.tight_layout()
plt.show()
