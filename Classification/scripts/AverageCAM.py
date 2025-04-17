print("##########################################################", flush=True)
print("0. Import Libraries and Mount Drive", flush=True)
import cv2,os,argparse
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from torchvision.models import resnet50
from Scripts.utils import ellipse, rectangle, rotate, ZeroPaddingResizeCV
print("done", flush=True)
print("##########################################################", flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stain_type", type=str, default="H3K27ac")
    parser.add_argument("--ctrl_type", type=str, default="RETT")
    parser.add_argument("--model_type", type=str, default="Resnet10_noavg")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--model_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results")
    parser.add_argument("--cam_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results_cam")
    parser.add_argument("--cam_type", type=str, default="GradCAM")
    args = parser.parse_args()
stain_type = args.stain_type
ctrl_type = args.ctrl_type
model_type = args.model_type
image_path = args.image_path
model_path = args.model_path
cam_path = args.cam_path
cam_type = args.cam_type
# 1. Define GradCAM
print("1. Define GradCAM", flush=True)
def predict(model,input_tensor,true_y):
    model.eval()
    output = model(input_tensor)
    pre_y = output.argmax(1).cpu().item()
    return pre_y==true_y, pre_y

def gradcams(model,input_tensor,target_layers,imgpad,cam_type):
    # GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    if cam_type=="GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="ScoreCAM":
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="AblationCAM":
        cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="XGradCAM":
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="EigenCAM":
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="FullGrad":
        cam = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
    else:
        return print("your option is not support")
    targets = None
    grayscale_cam = cam(input_tensor, targets)[0]
    visualization = show_cam_on_image(imgpad, grayscale_cam, use_rgb=False)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, targets)
    if np.amax(gb)!=0:
        gb = np.maximum(gb,0)*(1/np.amax(gb))
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = cam_mask * gb
    return visualization,gb,cam_gb,cam_mask

print("done", flush=True)
print("##########################################################", flush=True)

# 2. Load Data and Model
print("2. Load Data and Model", flush=True)
X = np.load(f"{image_path}/{ctrl_type}_{stain_type}.npy", allow_pickle=True)

fold = 0
loadmodel = f"{model_path}/{stain_type}_{model_type}/{stain_type}_{model_type}_Fold{str(fold)}.pkl"
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
device = "cuda"    
model = ResNet().to(device)
print("done", flush=True)
print("##########################################################", flush=True)

total = 3000
# 3. Average Gradcam heatmap in all data

print("3. Average Gradcam heatmap ", flush=True)
if ctrl_type == "CTRL": true_y = 0
elif ctrl_type == "RETT": true_y = 1
count = 0
intensity_conv1 = []
intensity_layer1 = []
intensity_layer2 = []
intensity_layer3 = []
intensity_layer4 = []
intensity_gb = []
for n in tqdm(range(total)):
    img = X[n]
    result, angle = ellipse(img)
    x, y, w, h = rectangle(result)
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        count+=1
        # conv1
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.conv1],img,cam_type)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(500, 500),n=1)
#         heatmap[heatmap<0.5]=0
        intensity_conv1.append(heatmap)
        # layer1
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer1],img,cam_type)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(500, 500),n=1)
#         heatmap[heatmap<0.5]=0
        intensity_layer1.append(heatmap)
        # layer2
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer2],img,cam_type)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(500, 500),n=1)
#         heatmap[heatmap<0.5]=0
        intensity_layer2.append(heatmap)
        # gb
        heatmap = ZeroPaddingResizeCV(rotate(gb[:,:,0], angle-45)[y:y+h,x:x+w], size=(500, 500),n=1)
#         heatmap[heatmap<0.5]=0
        intensity_gb.append(heatmap)
intensity_conv1 = np.array(intensity_conv1)
intensity_layer1 = np.array(intensity_layer1)
intensity_layer2 = np.array(intensity_layer2)
intensity_gb = np.array(intensity_gb)
print("intensity_layer2.shape: ", intensity_layer2.shape, flush=True)
print("Data acc: {:.3f}, count: {:}, total: {:} " .format(count/total,count,total), flush=True)
savename = f"{cam_type}_{ctrl_type}_{stain_type}"
np.save(f"{cam_path}/{savename}_conv1.npy" ,intensity_conv1)
np.save(f"{cam_path}/{savename}_layer1.npy" ,intensity_layer1)
np.save(f"{cam_path}/{savename}_layer2.npy" ,intensity_layer2)
np.save(f"{cam_path}/{savename}_gb.npy" ,intensity_gb)
print(f"Save all cam heatmap to {cam_path}/{savename} as npy", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)