import torch
import torch.nn as nn
import timm

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expects (Batch, Seq, Features)
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_output, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + self.dropout(attn_output)

        x_norm2 = self.norm2(x)
        ff_output = self.feed_forward(x_norm2)
        x = x + self.dropout(ff_output)

        return x
    


class XceptionTransformer(nn.Module):
    def __init__(self, output_size, embed_dim=2048, num_heads=8, num_layers=2, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.xception_base = timm.create_model(
            'xception',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        for param in self.xception_base.parameters():
            param.requires_grad = False

        self.sequence_len = 100
        self.positional_embedding = nn.Parameter(torch.randn(1, self.sequence_len, embed_dim))

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim), # Normalize the final output
            nn.Linear(embed_dim, output_size)
        )

    def forward(self, x):
        x = self.xception_base(x)
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier_head(x)

        return x