"""
Minimal Vision-Language Model, built from scratch.

Three components, matching the architecture pattern covered in Week 1's
literature review (CLIP -> Flamingo -> BLIP-2 -> LLaVA):

    1. PatchEmbedding + ViTEncoder   -- a small Vision Transformer, trained
       from scratch (no pretrained CLIP weights - the point of this
       exercise is to understand what a vision encoder actually does).
    2. Projector                     -- a linear layer mapping visual
       embeddings into the text decoder's embedding space (LLaVA-style:
       the simplest possible connector).
    3. TextDecoder                   -- a small causal Transformer decoder
       (a "mini GPT") that treats projected visual tokens as a prefix,
       then generates caption tokens autoregressively.

This is intentionally small-scale (a few million parameters, not billions)
so it trains in minutes on a laptop CPU/GPU. The goal is understanding the
mechanics, not benchmark performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Vision encoder (ViT), trained from scratch
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Splits an image into fixed-size patches and linearly embeds each one.

    This is the very first step of any ViT: turn a (C, H, W) image into a
    sequence of (num_patches, embed_dim) tokens, the same way a tokenizer
    turns a sentence into a sequence of token embeddings.
    """

    def __init__(self, img_size=96, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        # A single strided conv is the standard trick to implement
        # "split into patches + linearly project each patch" in one op.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attn_out)


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block: self-attn + MLP, each
    with a residual connection. This is the repeating unit that both the
    ViT encoder and the text decoder are built from."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """A minimal Vision Transformer encoder.

    Mirrors the architecture from the original ViT paper (patch embedding +
    learnable [CLS]/position embeddings + stack of Transformer blocks), at
    a much smaller scale than any production VLM's vision encoder.
    """

    def __init__(self, img_size=96, patch_size=16, in_channels=3,
                 embed_dim=192, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                                   # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                      # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # (B, N+1, D) -- includes the [CLS] token at position 0


# ---------------------------------------------------------------------------
# 2. Projector -- bridges vision embedding space to text embedding space
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """The simplest possible vision-language connector (LLaVA-style):
    a single linear layer. Contrast this with BLIP-2's Q-Former or
    Flamingo's Perceiver Resampler + gated cross-attention -- this is
    deliberately the minimal version so the mechanics are transparent."""

    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# 3. Text decoder -- a small causal Transformer ("mini GPT")
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Same as MultiHeadSelfAttention but with a causal mask, so each
    token can only attend to itself and earlier tokens -- required for
    autoregressive text generation."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attn_out)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TextDecoder(nn.Module):
    """A small causal Transformer decoder ("mini GPT") that generates text
    conditioned on a prefix of visual tokens.

    Visual tokens (already projected into this same embedding space) are
    simply prepended to the text token sequence -- the same "treat vision
    as a prefix" idea LLaVA uses, just without a pretrained LLM backbone.
    """

    def __init__(self, vocab_size, max_seq_len, embed_dim=192, depth=4,
                 num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, visual_tokens, text_token_ids):
        """
        visual_tokens:   (B, V, D)  -- already-projected visual embeddings
        text_token_ids:  (B, T)     -- integer token ids for the caption
        returns logits over the vocabulary for each text position, shape (B, T, vocab_size)
        """
        B, T = text_token_ids.shape
        text_embed = self.token_embed(text_token_ids)               # (B, T, D)
        x = torch.cat([visual_tokens, text_embed], dim=1)            # (B, V+T, D)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        # We only care about predictions made *at text positions*, i.e. the
        # last T positions (each predicting the *next* token).
        V = visual_tokens.shape[1]
        return logits[:, V - 1: V - 1 + T, :]  # align so logits[:, i] predicts text_token_ids[:, i]


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------

class MiniVLM(nn.Module):
    def __init__(self, vocab_size, max_text_len,
                 img_size=96, patch_size=16, embed_dim=192,
                 vit_depth=4, decoder_depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.vision_encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=vit_depth, num_heads=num_heads, dropout=dropout,
        )
        num_visual_tokens = self.vision_encoder.patch_embed.num_patches + 1  # +1 for [CLS]
        self.projector = Projector(embed_dim, embed_dim)
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            max_seq_len=num_visual_tokens + max_text_len,
            embed_dim=embed_dim, depth=decoder_depth, num_heads=num_heads, dropout=dropout,
        )

    def forward(self, images, text_token_ids):
        vision_features = self.vision_encoder(images)          # (B, V, D)
        visual_tokens = self.projector(vision_features)         # (B, V, D)
        logits = self.text_decoder(visual_tokens, text_token_ids)
        return logits

    @torch.no_grad()
    def generate(self, image, bos_id, eos_id, max_new_tokens=30):
        """Greedy autoregressive caption generation for a single image."""
        self.eval()
        device = image.device
        vision_features = self.vision_encoder(image.unsqueeze(0))
        visual_tokens = self.projector(vision_features)
        generated = torch.tensor([[bos_id]], device=device)
        for _ in range(max_new_tokens):
            logits = self.text_decoder(visual_tokens, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return generated.squeeze(0)
