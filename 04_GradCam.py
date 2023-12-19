import sys
sys.path.append("../pyDeepInsight/")
import numpy as np
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot

import pyDeepInsight.image_transformer
from sklearn.datasets import make_blobs

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

import warnings;
warnings.simplefilter('ignore')

# load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = torch.hub.load('pytorch/vision:v0.10.0',
                     'squeezenet1_1',
                     pretrained = True,
                     verbose = False)

num_classes = 2 # binominal 0/1 -> 2
net.classifier[1] = nn.Conv2d(512,
                              num_classes,
                              kernel_size = (1, 1),
                              stride = (1, 1))

# Transfer model to GPU/CPU
net = net.to(device)
net.load_state_dict(torch.load("models/last_model.pth",
                               map_location = torch.device('cpu')))

# load X_train/test_img and y_train/test
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

X_train_img = np.load('outputs/X_train_img.npy')
X_test_img = np.load('outputs/X_test_img.npy')

with open('outputs/y_train.pkl', "rb") as fh:
  y_train = pickle.load(fh)
with open('outputs/y_test.pkl', "rb") as fh:
  y_test = pickle.load(fh)

preprocess = transforms.Compose([
    transforms.ToTensor()
])

le = LabelEncoder()

y_train = le.fit_transform(y_train) 
y_test = le.transform(y_test)
le_mapping = dict(zip(le.transform(le.classes_),
                      le.classes_))

X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

# load the image
import pickle
with open('models/ImageTransformer_tSNE.pkl', "rb") as fh:
  it = pickle.load(fh)

# Deep Feature: CAM-based feature selection
# CAM, Class Activation Mapping
from pyDeepInsight import CAMFeatureSelector

## Step 1 - CAMFeatureSelector object
cm_method = 'GradCAM'
camfs = CAMFeatureSelector(model = net,
                           it = it,
                           cam_method = cm_method)

## Step 2 - Compute Class-Specific CAMs

fl_method = "max"
class_cam = camfs.calculate_class_activations(X_train_tensor,
                                              y_train_tensor,
                                              batch_size = 100,
                                              flatten_method = fl_method)

## Step 3 - Select Class-Specific Features
fs_threshold = 0.2
feat_idx = camfs.select_class_features(cams = class_cam,
                                       threshold = fs_threshold)

df_class_cam0 = pd.DataFrame(class_cam[0])
df_class_cam1 = pd.DataFrame(class_cam[1])
y = y_train_tensor.detach().cpu().numpy()

cat = np.unique(y)[1]
np.where(y == cat)
np.where(y == cat)[0]
cat_idx = np.where(y == cat)[0]
X = X_train_tensor
X[cat_idx, :, :, :].detach().mean(dim=0).cpu().numpy()

cam = class_cam

# Create image

from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt

def cam_image(X, y, cam, fs, threshold):
    fig, axs = plt.subplots(ncols = 2,
                            nrows = 1,
                            figsize = (2, 1),
                            constrained_layout = True,
                            squeeze = False) 

    for cat in np.unique(y):
        row = cat // 2 
        col = cat
        cat_idx = np.where(y == cat)[0] 
        X_cat = X[cat_idx, :, :, :].detach().mean(dim=0).cpu().numpy()
        cam_cat = cam[cat].copy()
        cam_cat[cam_cat <= threshold] = 0

        visualization = show_cam_on_image(np.transpose(X_cat, (1, 2, 0)),
                                          cam_cat,
                                          use_rgb = True)
        _ = axs[row, col].imshow(visualization)

        axs[row, col].text(0, 0, le_mapping[cat],
                           c = "white",
                           ha = "left",
                           va = "top",
                           weight = "bold",
                           size = "x-large")

        axs[row, col].text(227, 227, f"{fs[cat].shape[0]} genes",
                           c = "white",
                           ha = "right",
                           va = "bottom",
                           weight = "bold",
                           size = "large")
        axs[row, col].axis('off')
    return fig, axs
  
  fig, _ = cam_image(X_train_tensor,
              y_train_tensor.detach().cpu().numpy(),
              class_cam,
              feat_idx,
              fs_threshold)

fig.set_size_inches(18.5, 10.5)
fig.savefig("gradcam/overlay.png", dpi = 300)

coords = it.coords() # 座標情報を取得

df_all = pd.DataFrame(np.array(coords),
                      columns = ['coord0', 'coord1'])
df_all['feature'] = np.array(feature_data.columns)
df_all.to_csv('gradcam/Coords_IT.csv',
          index = False)
