import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.resnet.fc = nn.Linear(512, 2)   
    def forward(self, x):
        x = self.resnet(x)
        # x = nn.Softmax(dim=1)(x)
        return x