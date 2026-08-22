import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class Hubert(nn.Module):
    def __init__(self, num_label_embeddings=100, mask=True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(nn.TransformerEncoderLayer(768, 12, 3072, activation="gelu", batch_first=True), 12)
        self.proj = nn.Linear(768, 256)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x):  # SpecAugment spans (training only; inference keeps mask=None)
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(self, x, layer=None):
        x = self.feature_projection(self.feature_extractor(x).transpose(1, 2))
        x, mask = self.mask(x)
        x = self.dropout(self.norm(x + self.positional_embedding(x)))
        return self.encoder(x, output_layer=layer), mask

    def logits(self, x):  # cosine-sim / temperature 0.1
        return torch.cosine_similarity(x.unsqueeze(2), self.label_embedding.weight.unsqueeze(0).unsqueeze(0), dim=-1) / 0.1

    def forward(self, x):
        x, mask = self.encode(x)
        return self.logits(self.proj(x)), mask

class HubertSoft(Hubert):
    @torch.inference_mode()
    def units(self, wav):  # pad (400-320)//2 = 40 samples/side, hubert-soft alignment convention
        x, _ = self.encode(F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2)))
        return self.proj(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1, self.conv2, self.conv3, self.conv4 = [nn.Conv1d(512, 512, 3, 2, bias=False) for _ in range(4)]
        self.conv5, self.conv6 = [nn.Conv1d(512, 512, 2, 2, bias=False) for _ in range(2)]

    def forward(self, x):
        x = F.gelu(self.norm0(self.conv0(x)))
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6):
            x = F.gelu(conv(x))
        return x

class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): return self.dropout(self.projection(self.norm(x)))

class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(768, 768, kernel_size=128, padding=128 // 2, groups=16), name="weight", dim=2)

    def forward(self, x):  # even kernel -> sym pad leaves len+1; drop right sample
        return F.gelu(self.conv(x.transpose(1, 2))[:, :, :-1]).transpose(1, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, output_layer=None):
        for layer in self.layers[:output_layer]:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src

def _compute_mask(shape, mask_prob, mask_length, device, min_masks=0):
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")
    if mask_length > sequence_length:
        raise ValueError(f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`")
    num_masked_spans = max(int(mask_prob * sequence_length / mask_length + random.random()), min_masks)
    num_masked_spans = min(num_masked_spans, sequence_length // mask_length)
    spans = torch.multinomial(torch.ones((batch_size, sequence_length - (mask_length - 1)), device=device), num_masked_spans)
    mask_idxs = (spans.unsqueeze(-1) + torch.arange(mask_length, device=device)[None, None, :]).reshape(batch_size, -1)
    return torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool).scatter(1, mask_idxs, True)