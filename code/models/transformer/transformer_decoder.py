import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        heads=8,
        layers=3,
        max_len=200
    ):

        super().__init__()


        self.embed_dim = embed_dim


        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )


        # Positional embedding for LaTeX sequence
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                embed_dim
            )
        )


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=heads,
            batch_first=True
        )


        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=layers
        )


        # Convert hidden states -> vocabulary probabilities
        self.fc = nn.Linear(
            embed_dim,
            vocab_size
        )


    def generate_square_subsequent_mask(self, size, device):

        """
        Causal mask.

        Prevents token i from seeing future tokens.

        Example:

        token 1 can see:
        token 1

        token 2 can see:
        token 1, token 2

        token 3 can see:
        token 1, token 2, token 3
        """

        mask = torch.triu(
            torch.ones(size, size, device=device),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        return mask



    def forward(
        self,
        image_features,
        captions,
        padding_mask=None
    ):

        """
        image_features:
            (B, num_image_tokens, embed_dim)

        captions:
            (B, sequence_length)

        padding_mask:
            (B, sequence_length)
            True = ignore token
        """


        B,T = captions.shape


        device = captions.device


        # Token embedding
        x = self.embedding(
            captions
        )


        # Add positional encoding
        x = x + self.pos_embedding[:, :T, :]



        # Prevent looking ahead
        causal_mask = self.generate_square_subsequent_mask(
            T,
            device
        )



        # Transformer decoder

        output = self.transformer(
            tgt=x,
            memory=image_features,

            # future token mask
            tgt_mask=causal_mask,

            # ignore PAD tokens
            tgt_key_padding_mask=padding_mask
        )



        # Vocabulary prediction

        output = self.fc(
            output
        )


        return output