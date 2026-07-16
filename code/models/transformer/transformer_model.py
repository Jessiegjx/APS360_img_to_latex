import torch.nn as nn

from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder


class ImageToLatexTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx
    ):

        super().__init__()

        self.encoder = ResNetEncoder(
            embed_dim=256
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=256,
            heads=8,
            layers=3,
            pad_idx=pad_idx
        )


    def forward(self, images, captions):

        features = self.encoder(images)

        outputs = self.decoder(
            features,
            captions
        )

        return outputs