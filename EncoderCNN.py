from torchvision import models
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        efficientnet = models.efficientnet_b3(pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad_(False)

        self.efficientnet = nn.Sequential(*list(efficientnet.children())[:-2])

    def forward(self, images):
        features = self.efficientnet(images)                            # (batch_size, 1536, 7, 7)
        features = features.permute(0, 2, 3, 1)                         # (batch_size, 7, 7, 1536)
        features = features.view(features.size(0), -1, features.size(-1)) # (batch_size, 49, 1536)
        return features

