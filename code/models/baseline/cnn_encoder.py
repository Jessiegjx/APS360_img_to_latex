import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, embed_size)
        # self.fc = nn.Sequential(
        #     nn.Linear(128, embed_size),
        #     nn.Tanh()
        # )


    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
