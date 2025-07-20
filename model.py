import torch.nn as nn
from torchvision import models

class ASLClassifier(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLClassifier, self).__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
