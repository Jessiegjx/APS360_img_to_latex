import torch
import torch.nn as nn

# RNN
class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=256
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )


    def forward(self,features,captions):
        embeddings = self.embedding(captions)
        # image feature becomes first hidden state
        h0 = features.unsqueeze(0)
        outputs,_ = self.rnn(
            embeddings,
            h0
        )
        outputs = self.fc(outputs)
        return outputs