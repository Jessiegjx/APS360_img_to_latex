import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        resnet=resnet152(weights=ResNet152_Weights.DEFAULT)
        self.backbone=nn.Sequential(*list(resnet.children())[:-2])
        # self.backbone=nn.Sequential(*list(resnet.children())[:-3]) # more visual tokens at once.

        self.projection=nn.Conv2d(
            2048,
            embed_dim,
            kernel_size=1
        )


    def positional_encoding_2d(self,C,H,W,device):
        pe=torch.zeros(C,H,W,device=device)
        c=C//4
        y_pos=torch.arange(H,device=device).unsqueeze(1)
        x_pos=torch.arange(W,device=device).unsqueeze(1)
        div=torch.exp(
            torch.arange(0,c,2,device=device)
            *
            (-torch.log(torch.tensor(10000.0,device=device))/c)
        )

        pe[0:c:2,:,:]=torch.sin(y_pos*div).T.unsqueeze(2).repeat(1,1,W)
        pe[1:c:2,:,:]=torch.cos(y_pos*div).T.unsqueeze(2).repeat(1,1,W)
        pe[c:2*c:2,:,:]=torch.sin(x_pos*div).T.unsqueeze(1).repeat(1,H,1)
        pe[c+1:2*c:2,:,:]=torch.cos(x_pos*div).T.unsqueeze(1).repeat(1,H,1)

        return pe.unsqueeze(0)


    def forward(self,x):
        x=self.backbone(x)
        x=self.projection(x)
        B,C,H,W=x.shape

        pe=self.positional_encoding_2d( C,H, W,x.device)

        x=x+pe
        x=x.flatten(2)
        x=x.permute(0,2,1)

        return x

'''
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=256, max_tokens=200):
        super().__init__()
        resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
      
        # remove avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, embed_dim, kernel_size=1)
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
        # print(x.mean().item(), x.std().item())
        return x
        '''