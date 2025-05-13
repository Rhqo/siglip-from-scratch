from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from torchvision import transforms

def preprocess_image(image, image_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image)
    # Add batch dimension (3, 224, 224) -> (1, 3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

@dataclass
class SigLipVisionConfig:
    # Embedding
    num_channels: int = 3
    embed_dim: int = 768
    image_size: int = 224
    patch_size: int = 16
    # EncoderBlock
    num_attention_heads: int = 12
    attention_dropout: float = 0.0
    # Encoder
    num_encoder_blocks: int = 12
    # MLP
    mlp_hidden_dim: int = 3072
    # LayerNorm
    layer_norm_eps: float = 1e-6

class VisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # register_buffer : training X, state maintain
        # self.position_ids = torch.arange(self.num_positions).expand((1, -1))
        self.register_buffer(
            "position_ids",
            # (1, 196)
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape

        # (3, 224, 224) -> (B, 3, 224, 224)
        patch_embeds = self.patch_embedding(pixel_values)
        # (B, 768, 14, 14) -> (B, 768, 196) -> (B, 196, 768)
        embeddings = patch_embeds.flatten(-2, -1).transpose(1, 2)
        # 196 patches, 768-dimension vector
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

class MultiheadAttention(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        # x : (batch_size, num_patches, embed_dim)
        # (B, 196, 768)
        B, T, C = x.shape

        # proj : (B, 196, 768)
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        # 12 heads, 196 patches, 64-dimension vector
        # states : (B, 196, 768) -> (B, 196, 12, 64) -> (B, 12, 196, 64)
        q_states = q_proj.reshape(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_states = k_proj.reshape(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_states = v_proj.reshape(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # (B, 12, 196, 64) @ (B, 12, 64, 196) -> (B, 12, 196, 196)
        attention_scores = q_states @ k_states.transpose(-2, -1) * (k_states.size(-1) ** -0.5)
        attention_probs = F.softmax(attention_scores, dim=-1).to(q_states.dtype)
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        # (B, 12, 196, 196) @ (B, 12, 196, 64) -> (B, 12, 196, 64)
        attention_out = attention_probs @ v_states
        # (B, 12, 196, 64) -> (B, 196, 12, 64) -> (B, 196, 768)
        attention_out = attention_out.transpose(1, 2).reshape(B, T, C)
        attention_out = self.out_proj(attention_out)

        return attention_out

class MLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_hidden_dim)
        self.fc2 = nn.Linear(config.mlp_hidden_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = MultiheadAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)     

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x

class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.num_encoder_blocks = config.num_encoder_blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_encoder_blocks)])

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x

# class SigLipPoolingHead(nn.Module):
    
class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = VisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        # (B, 3, 224, 224) -> (B, 196, 768)
        embeddings = self.embeddings(pixel_values)
        # (B, 196, 768) -> (B, 196, 768)
        encoder_out = self.encoder(embeddings)
        out = self.post_layernorm(encoder_out)

        return out
    
class SigLipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.transformer = SigLipVisionTransformer(config)

    def forward(self, pixel_values):
        # (B, 3, 224, 224) -> (B, 768)
        out = self.transformer(pixel_values)
        return out
