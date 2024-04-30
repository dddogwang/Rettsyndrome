print("🥁 PYTHON SCRIPT START", flush=True)
# 0. Import Libraries and Mount Drive
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import warnings
import logging
warnings.filterwarnings("ignore")
import os, cv2, sys, argparse
from skimage import io
from tqdm import tqdm
from scipy import stats
import shutil
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops
import tensorflow as tf
from keras.models import *
import keras
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from imgaug import augmenters as iaa
from datetime import timedelta
import datetime
import imageio
import math
import pytz
from pytz import timezone
import imagecodecs._imcd
from tifffile import imread

from root.Mask_RCNN.mrcnn.config import Config
from root.Mask_RCNN.mrcnn import utils
from root.Mask_RCNN.mrcnn import model as modellib
from root.Mask_RCNN.mrcnn import visualize
from root.Mask_RCNN.mrcnn import postprocessing
from root.Mask_RCNN.mrcnn.model import log
import nucleus
print("##########################################################", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagetype", type=str, default="ctrl")
    parser.add_argument("--imagepath", type=str, default="/home/acd13264yb/DDDog/Datasets/240420Rettsyndrome/deconv/ctrl/")
    parser.add_argument("--imagename", type=str, default="CTRL HiPSC_1_Dapi_488_CTCF_555H3K27AC006 - Deconvolved 3 iterations, Type Richardson-Lucy_XY001.tif")
    parser.add_argument("--save_segment_path", type=str, default="./results/")
    parser.add_argument("--save_cell_path", type=str, default="/home/acd13264yb/DDDog/Datasets/240420Rettsyndrome/SingeleCell/deconv/")
    parser.add_argument("--weightpath", type=str, default=None)
    args = parser.parse_args()

imagetype = args.imagetype
imagepath = args.imagepath
imagename = args.imagename
save_segment_path = args.save_segment_path
save_segment_name = f"{save_segment_path}/{imagename[:-4]}_segment.png"
save_cell_path = args.save_cell_path
weightpath = args.weightpath
print("imagepath is ",imagepath, flush=True)
print("imagename is ",imagename, flush=True)
print("save_segment_path is ",save_segment_path, flush=True)
print("save_segment_name is ",save_segment_name, flush=True)
print("save_cell_path    is ",save_cell_path, flush=True)


# 1. Load and Process Images
print("🚀 1. Load and Process Images", flush=True)
loadpath = f"{imagepath}/{imagename}"
img_origin = io.imread(loadpath)
b = img_origin[:,:,2]
img_dapi = cv2.merge([b,b,b])
img_dapi *= int(255/img_dapi.max())
print(f"Load Images {loadpath}", flush=True)
print("##########################################################", flush=True)


# 2. Configuration
print("🚀 2. Configuration", flush=True)
class NucleusInferenceConfig(nucleus.NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MAX_DIM = 512

    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.5
    RPN_NMS_THRESHOLD = 0.2
    
    BACKBONE = "resnet50"
    
    MEAN_PIXEL = np.mean(img_dapi,axis=(0,1))
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (100, 100)
print("##########################################################", flush=True)


# 3. Run the Network
print("🚀 3. Run the Network", flush=True)
Weights = "Kaggle"
if Weights == "Kaggle":
    weights_path = weightpath+'/mask_rcnn_kaggle_v1.h5' 
elif Weights == "Storm_Cell":
    weights_path = weightpath+'/mask_rcnn_nucleus_cell.h5' 
print("weights_path : ", weights_path)

config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_path)
model.load_weights(weights_path, by_name=True)
result = model.detect([img_dapi], verbose=0)[0]

mask, bbox, class_ids = result["masks"], result["rois"], result["class_ids"]
print("Origin shape: ", img_origin.shape, flush=True)
log("image", img_dapi)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
print("##########################################################", flush=True)


# 4. post processing
print("🚀 4. post processing", flush=True)
## 4.1 remove outliers by roundness
print("# 4.1 remove outliers by roundness", flush=True)
def compute_roundness(label_image):
    contours, hierarchy = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b

total_roundness=[]
for i in range(bbox.shape[0]):
    mmask = mask[:,:,i]
    total_roundness.append(compute_roundness(mmask))
print("len(total_roundness): ",len(total_roundness), flush=True)
roundness_outliers = []
roundness_zscore = np.abs(stats.zscore(total_roundness))
for c in range(bbox.shape[0]):
    roundness_outliers.append((total_roundness[c])<0.68)
print("roundness_outliers: ", flush=True)
for c in range(len(total_roundness)):
    if roundness_outliers[c]:
        print((roundness_outliers[c], c, total_roundness[c], roundness_zscore[c]), flush=True)

print("# 4.2 remove outliers by masks_area", flush=True)
total_masks_area=[]
for i in range(bbox.shape[0]):
    total_masks_area.append(np.sum(mask[:,:,i]))
print("len(total_masks_area): ",len(total_masks_area))
masks_outliers = []
masks_zscore = np.abs(stats.zscore(total_masks_area))
for c in range(bbox.shape[0]):
    masks_outliers.append(((total_masks_area[c])<20000) or (masks_zscore[c]>2.5))
print("masks_outliers: ", flush=True)
for c in range(len(total_masks_area)):
    if masks_outliers[c]:
        print((masks_outliers[c], c, total_masks_area[c], masks_zscore[c]), flush=True)

print("# 4.3 remove outliers by boxes_aspect", flush=True)
total_boxes_size=[]
for i in range(bbox.shape[0]):
    box = bbox[i]
    total_boxes_size.append((box[2]-box[0])/(box[3]-box[1]))
print("len(total_boxes_size): ",len(total_boxes_size))
boxes_outliers = []
boxes_zscore = np.abs(stats.zscore(total_boxes_size))
for zs in boxes_zscore:
    boxes_outliers.append((zs>=1.5))
print("boxes_outliers: ", flush=True)     
for c in range(len(total_boxes_size)):
    if boxes_outliers[c]:
        print((boxes_outliers[c], c, total_boxes_size[c], boxes_zscore[c]), flush=True)

print("# 4.4 All outliers together", flush=True)
outliers=[]
for i in range(len(masks_outliers)): 
    outliers.append(masks_outliers[i] or roundness_outliers[i] or boxes_outliers[i])
F_total_boxes=[]
F_total_masks=[]
for cc in range(len(outliers)):
    if not outliers[cc]:
        F_total_boxes.append(bbox[cc])
        F_total_masks.append(mask[:,:,cc])
F_total_boxes = np.array(F_total_boxes)
F_total_masks=np.transpose(np.array(F_total_masks),(1,2,0))
F_total_class_ids=np.ones(F_total_boxes.shape[0], dtype=np.int32)
print("total_boxes.shape: ",F_total_boxes.shape, flush=True)
print("total_masks.shape: ",F_total_masks.shape, flush=True)
print("visualize.display_instances", flush=True)
visualize.display_instances_save(img_dapi, F_total_boxes, F_total_masks, F_total_class_ids, ["BG","nucleus"], savename=save_segment_name, figsize=(20, 20))
print("save maskrcnn segmentation results as ", save_segment_name, flush=True)
print("##########################################################", flush=True)

# 5. Split and save each cell
print("🚀 5. Split and save each cell", flush=True)
for nn in range(F_total_boxes.shape[0]):
    bb = F_total_boxes[nn]
    mmask = F_total_masks[:,:,nn][bb[0]:bb[2], bb[1]:bb[3]].astype(np.uint8)
    mmask = cv2.merge([mmask,mmask,mmask])
    iimg = img_origin[bb[0]:bb[2], bb[1]:bb[3]]

    iimg = np.multiply(iimg, mmask)
    rr,gg,bb = cv2.split(iimg)
    save_cell_name = f"{imagename[:-4]}_single{nn:02}"
    cv2.imwrite(f"{save_cell_path}/All/{imagetype}/{save_cell_name}_All.tif", iimg)
    cv2.imwrite(f"{save_cell_path}/H3K27ac/{imagetype}/{save_cell_name}_H3K27AC.tif", rr)
    cv2.imwrite(f"{save_cell_path}/CTCF/{imagetype}/{save_cell_name}_CTCF.tif", gg)
    cv2.imwrite(f"{save_cell_path}/Dapi/{imagetype}/{save_cell_name}_Dapi.tif", bb)
    print("Save name with ", f"{save_cell_path}/All/{imagetype}/{save_cell_name}_All.tif", flush=True)