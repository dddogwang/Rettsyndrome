print("🥁 PYTHON SCRIPT START", flush=True)
# 0. Import Libraries and Mount Drive
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import argparse, sys
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
    parser.add_argument("--rett_type", type=str, default="HPS3042")
    parser.add_argument("--model_type", type=str, default="Resnet10_noavg")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    args = parser.parse_args()

stain_type = args.stain_type
rett_type = args.rett_type
model_type = args.model_type
image_path = args.image_path
save_path = args.save_path

# 1. Load and Process Images
print("🚀 1. Load and Process Images", flush=True)
X_Ctrl = np.load(f"{image_path}/CTRL_{stain_type}.npy",allow_pickle=True)
X_Rett = np.load(f"{image_path}/RETT_{rett_type}_{stain_type}.npy",allow_pickle=True)
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
sys.path.append('/home/acd13264yb/WorkSpace/Rettsyndrome/Classification')
from models.Resnet10_ldr import MyModel
from models.Resnet10_ldr import linear_dependency_regularization
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 3. Define Training and Validation", flush=True)
def train(model,device,dataloader_train,criterion,optimizer,ldr_weight=0.1):
    losses_train = []
    n_train = 0
    acc_train = 0
    optimizer.step()
    model.train()
    for x, y in dataloader_train:
        n_train += y.size()[0]
        model.zero_grad()  # 勾配の初期化
        x = x.to(device)  # テンソルをGPUに移動
        y = y.to(device)
        feature, output = model(x)  # 順伝播
        task_loss = criterion(output, y)  # 誤差(クロスエントロピー誤差関数)の計算
        ldr_loss = linear_dependency_regularization(feature)
        loss = task_loss + ldr_weight*ldr_loss
        loss.backward()  # 誤差の逆伝播
        optimizer.step()  # パラメータの更新
        acc_train += (output.argmax(1) == y[:,1]).float().sum().item()
        losses_train.append(loss.tolist())
    return np.mean(losses_train), (acc_train/n_train)       
def valid(model,device,dataloader_valid,criterion,ldr_weight=0.1):
    losses_valid = []
    n_val = 0
    acc_val = 0
    model.eval()
    for x, y in dataloader_valid:
        n_val += y.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        y = y.to(device)
        feature, output = model(x)  # 順伝播
        task_loss = criterion(output, y)  # 誤差(クロスエントロピー誤差関数)の計算
        ldr_loss = linear_dependency_regularization(feature)
        loss = task_loss + ldr_weight*ldr_loss
        acc_val += (output.argmax(1) == y[:,1]).float().sum().item()
        losses_valid.append(loss.tolist())
    return np.mean(losses_valid), (acc_val/n_val)
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 4. Train by KFold of Cross Validation", flush=True)
n_splits=5
splits=KFold(n_splits,shuffle=True,random_state=42)
batch_size = args.batch_size
n_epochs = args.n_epochs
history = {'loss_train': [], 'loss_valid': [],'acc_train':[],'acc_valid':[]}
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1), flush=True)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    dataloader_valid = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    model = MyModel().to(device)
    ngpu = 4
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    weights = torch.tensor([(len(X_Ctrl)+len(X_Rett))/len(X_Ctrl), 
                            (len(X_Ctrl)+len(X_Rett))/len(X_Rett)]).cuda()
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for epoch in range(n_epochs):
        loss_train, acc_train = train(model,device,dataloader_train,criterion,optimizer)
        loss_valid, acc_valid = valid(model,device,dataloader_valid,criterion)
        scheduler.step()
        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'
              .format(epoch, loss_train, acc_train, loss_valid, acc_valid), flush=True)
        history['loss_train'].append(loss_train)
        history['loss_valid'].append(loss_valid)
        history['acc_train'].append(acc_train)
        history['acc_valid'].append(acc_valid)
    savemodel = f"{save_path}/{rett_type}_{stain_type}_{model_type}_Fold{str(fold)}.pkl"
    for param in model.parameters():
        param.requires_grad = True
        torch.save(model.module.state_dict(),savemodel)
    print("🏵️ saved model as "+savemodel, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 5. plot the figure of training process", flush=True)
loss_train_avg = np.zeros(n_epochs)
loss_valid_avg = np.zeros(n_epochs)
acc_train_avg = np.zeros(n_epochs)
acc_valid_avg = np.zeros(n_epochs)
for num in range(n_splits):
    loss_train = history['loss_train'][num*n_epochs:(num+1)*n_epochs]
    loss_valid = history['loss_valid'][num*n_epochs:(num+1)*n_epochs]
    acc_train = history['acc_train'][num*n_epochs:(num+1)*n_epochs]
    acc_valid = history['acc_valid'][num*n_epochs:(num+1)*n_epochs]
    loss_train_avg+=loss_train
    loss_valid_avg+=loss_valid
    acc_train_avg+=acc_train
    acc_valid_avg+=acc_valid
    name_title="Fold_"+str(num)+'_Training and Validation accuracy'
    savepath = f"{save_path}/{name_title}.png"
    plt.figure(figsize=(12, 8))
    plt.ylim(0,1.0)
    plt.plot(range(1,n_epochs+1), acc_train, 'b', label='Training accuracy')  
    plt.plot(range(1,n_epochs+1), acc_valid, 'r', label='Validation accuracy')
    plt.title(name_title)
    plt.savefig(savepath)
    print("🌺 Saved output as , ", savepath, flush=True)
loss_train_avg = loss_train_avg/n_splits
loss_valid_avg = loss_valid_avg/n_splits
acc_train_avg = acc_train_avg/n_splits
acc_valid_avg = acc_valid_avg/n_splits
name_title="Average Training and Validation accuracy"
savepath = f"{save_path}/{name_title}.png"
plt.figure(figsize=(12, 8))
plt.ylim(0,1.0)
plt.plot(range(1,n_epochs+1), acc_train_avg, 'b', label='Training accuracy')  
plt.plot(range(1,n_epochs+1), acc_valid_avg, 'r', label='Validation accuracy')
plt.title(name_title)
plt.savefig(savepath)
print("🌸 Saved output as , ", savepath, flush=True)
print("##########################################################", flush=True)
print(history, flush=True)
print("done", flush=True)