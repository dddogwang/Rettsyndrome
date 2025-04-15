import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.basemodel = models.resnet18(weights=True)
        self.basemodel.layer3 = nn.Identity()
        self.basemodel.layer4 = nn.Identity()
        self.basemodel.fc = nn.Identity()
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        feature = self.basemodel(x)
        output = self.classifier(feature)
        return feature, output

# Linear Dependency Regularization
def feature_decorrelation(features):
    """
    features: Tensor, shape [batch_size, feature_dim]
    Calculates the linear dependency regularization loss based on the covariance matrix.
    """
    batch_size, feature_dim = features.size()
    cov_matrix = torch.mm(features.T, features) / batch_size  # Covariance matrix
    identity = torch.eye(feature_dim).to(features.device)  # Identity matrix
    decorr_loss = torch.norm(cov_matrix - identity, p='fro')**2  # Frobenius norm
    decorr_loss = decorr_loss / feature_dim  # 特征维度归一化
    return decorr_loss