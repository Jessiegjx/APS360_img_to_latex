import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNetEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        max_tokens=200
    ):
        super().__init__()

        resnet = resnet152(
            weights=ResNet152_Weights.DEFAULT
        )
      

        # remove avgpool and fc
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )


        self.projection = nn.Conv2d(
            2048,
            embed_dim,
            kernel_size=1
        )


        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_tokens,
                embed_dim
            )
        )


    def forward(self,x):

        x = self.backbone(x)

        x = self.projection(x)

        B,C,H,W = x.shape


        # B,C,H,W -> B,N,C
        x = x.flatten(2)
        x = x.permute(0,2,1)


        x = x + self.pos_embedding[:,:x.size(1),:]

        return x