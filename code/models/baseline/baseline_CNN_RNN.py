import torch.nn as nn

from models.baseline.cnn_encoder import CNNEncoder
from models.baseline.rnn_decoder import RNNDecoder


class CNN_RNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder = CNNEncoder(embed_size=256)
        self.decoder = RNNDecoder(vocab_size, embed_size=256, hidden_size=256)

    def forward(self,images,captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs