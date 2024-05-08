print("🥁 PYTHON SCRIPT START", flush=True)
# 0. Import Libraries and Mount Drive
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import cv2,os,argparse
from skimage import io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,precision_score,accuracy_score,roc_curve
import torch
from torch.utils.data import random_split,Dataset,DataLoader,SubsetRandomSampler
from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("torch.device(cuda)", flush=True)
    print("torch.cuda.device_count(): ", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(), flush=True)
    print("torch.cuda.current_device()", torch.cuda.current_device(), flush=True)
else:
    device = torch.device("cpu")
    print("torch.device(cpu)", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stain_type", type=str, default="H3K27ac")
    parser.add_argument("--model_type", type=str, default="Resnet10_noavg")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    args = parser.parse_args()

stain_type = args.stain_type
model_type = args.model_type
image_path = args.image_path
save_path = args.save_path

# 1. Load and Process Images
print("🚀 1. Load and Process Images", flush=True)
X_Ctrl = np.load(f"{image_path}/CTRL_{stain_type}.npy",allow_pickle=True)
X_Rett = np.load(f"{image_path}/RETT_{stain_type}.npy",allow_pickle=True)
y_Ctrl = torch.zeros(len(X_Ctrl), dtype=torch.int64)
y_Rett = torch.ones(len(X_Rett), dtype=torch.int64)
X = np.concatenate((X_Ctrl, X_Rett), axis = 0)
y = torch.cat((y_Ctrl, y_Rett), 0)
class cell_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.transform(self.x[idx]).to(torch.float), F.one_hot(self.y[idx],num_classes=2).to(torch.float)
dataset = cell_dataset(X, y)
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 2. Develop model", flush=True)
if model_type=="Resnet10_noavg":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.layer3 = nn.Sequential()
            self.resnet.layer4 = nn.Sequential()
            self.resnet.avgpool = nn.Sequential()
            self.resnet.fc = nn.Linear(128*63*63, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))
        def forward(self, x):
            x = self.resnet(x)
            # x = nn.Softmax(dim=1)(x)
            return x
elif model_type=="Resnet10":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.layer3 = nn.Sequential()
            self.resnet.layer4 = nn.Sequential()
            self.resnet.fc = nn.Linear(128, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))
        def forward(self, x):
            x = self.resnet(x)
            # x = nn.Softmax(dim=1)(x)
            return x
elif model_type=="Resnet18":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.fc = nn.Linear(512, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))
        def forward(self, x):
            x = self.resnet(x)
            # x = nn.Softmax(dim=1)(x)
            return x     
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 3. Define Training and Validation", flush=True)   
# 定义函数以计算和保存ROC曲线
def plot_and_save_roc_curve(y_true, y_scores, fold):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold+1} ROC')
    plt.legend(loc="lower right")
    name_title="Fold_"+str(fold)+'_Validation ROC'
    savepath = f"{save_path}/{name_title}.png"
    plt.savefig(savepath)
    plt.close()
    print("🌺 Saved output as , ", savepath, flush=True)
    return auc_score

# 你的valid函数现在需要收集预测的分数和实际标签，以计算AUC
def valid(model, device, dataloader_valid, loss_function):
    model.eval()
    y_true = []
    y_scores = []
    losses_valid = []
    acc_val = 0
    n_val = 0
    for x, y in dataloader_valid:
        n_val += y.size()[0]
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            output = model(x)
        y_true.extend(y[:,1].tolist())  # 假设y的第二列是标签
        y_scores.extend(output[:,1].sigmoid().tolist())  # 假设模型的第二个输出是预测为正类的得分
        loss = loss_function(output, y)
        acc_val += (output.argmax(1) == y[:,1]).float().sum().item()
        losses_valid.append(loss.item())
    auc_score = plot_and_save_roc_curve(y_true, y_scores, fold)  # 调用ROC绘图函数
    return np.mean(losses_valid), acc_val / n_val, auc_score
print("##########################################################", flush=True)

print("🚀 4. Train by KFold of Cross Validation", flush=True)
n_splits=5
splits=KFold(n_splits,shuffle=True,random_state=42)
batch_size = args.batch_size
n_epochs = args.n_epochs
history = {'acc_valid':[], 'loss_valid':[], 'auc_valid':[]}
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1), flush=True)
    valid_sampler = SubsetRandomSampler(val_idx)
    dataloader_valid = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    loadmodel = f"{save_path}/{stain_type}_{model_type}_Fold{str(fold)}.pkl"
    print("load model", flush=True)
    model = ResNet().to(device)
    ngpu = 1
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    weights = torch.tensor([(len(X_Ctrl)+len(X_Rett))/len(X_Ctrl), 
                            (len(X_Ctrl)+len(X_Rett))/len(X_Rett)]).cuda()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    loss_valid, acc_valid, auc_valid = valid(model, device, dataloader_valid, loss_function)
    print(f'Fold {fold+1}, EPOCH: [Valid [Loss: {loss_valid:.3f}, Accuracy: {acc_valid:.3f}, AUC: {auc_valid:.3f}]')
    history['loss_valid'].append(loss_valid)
    history['acc_valid'].append(acc_valid)
    history['auc_valid'].append(auc_valid)

print(history, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
