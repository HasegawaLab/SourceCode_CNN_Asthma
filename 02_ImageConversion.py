# 0. Preparation
import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE

from sklearn.manifold import TSNE

from pyDeepInsight import ImageTransformer
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# import dataset
feature_data = pd.read_csv('ClinicalmRNALipidProduct_Value.csv')
X = feature_data

asthma_data = pd.read_csv('ClinicalmRNALipidProduct_Asthma.csv') # normalized_counts
y = asthma_data['asthma_epi_6yr']

# make train/test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 24,
                                                    stratify = y)
# SMOTE
sm = SMOTE(random_state = 42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

X_train_sm = X_train_sm.fillna(0)
X_test = X_test.fillna(0)

# Label encode and Mapping
le = LabelEncoder() 
y_train_enc = le.fit_transform(y_train_sm)
y_test_enc = le.transform(y_test)

le_mapping = dict(zip(le.transform(le.classes_),
                      le.classes_))
num_classes = np.unique(y_train_enc).size

# tSNE
distance_metric = 'cosine'

reducer = TSNE(n_components = 2,
               metric = distance_metric,
               init = 'random',
               learning_rate = 'auto',
               n_jobs = -1,
               verbose = True,
               random_state = 42)

# Image Transform
pixel_size = (227, 227)
it = ImageTransformer(feature_extractor = reducer,
                      pixels = pixel_size)

it.fit(X_train_sm,
       y = y_train_sm,
       plot = True)

X_train_img = it.transform(X_train_sm)
X_test_img = it.transform(X_test)

pd.to_pickle(it,
             'models/ImageTransformer_tSNE.pkl')

# Convert images to ones with colors
clinical_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'Clinical__' in col]
omics1_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'Omics1__' in col and 'Clinical' not in col]
omics2_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'Omics2__' in col and 'Clinical' not in col]

clinicalomics1_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'ClinicalOmics1__' in col]
clinicalomics2_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'ClinicalOmics2__' in col]
omics1omics2_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'Dualomics__' in col and 'Clinical' not in col]
clinicalomics1omics2_columns = [idx for idx, col in enumerate(X_train_sm.columns) if 'ClinicalDualomics__' in col]

coords = it.coords()

X_img = X_train_img 
new_img = np.zeros_like(X_img) 
## train data
for k in range(0, X_train_sm.shape[1]-1): # k = 0–3219
    x = coords[k][0]
    y = coords[k][1]

    if k in clinical_columns: # Red: Virus
        new_img[:, x, y, 0] = X_img[:, x, y, 0]

    elif k in omics1_columns: # Green: mRNA
        new_img[:, x, y, 1] = X_img[:, x, y, 1]

    elif k in omics2_columns: # Green: lipid
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    elif k in clinicalomics1_columns: # Yellow (RG)
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 1] = X_img[:, x, y, 1]

    elif k in clinicalomics2_columns: # Pink (RB)
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    elif k in omics1omics2_columns: # Skyblue (GB)
        new_img[:, x, y, 1] = X_img[:, x, y, 1]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    else: # RGB: virusmrnalipid
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 1] = X_img[:, x, y, 1]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

X_train_img = new_img

## test data
X_img = X_test_img
new_img = np.zeros_like(X_img)

for k in range(0, X_test.shape[1]-1):
    x = coords[k][0]
    y = coords[k][1]

    if k in clinical_columns: # Red: Virus
        new_img[:, x, y, 0] = X_img[:, x, y, 0]

    elif k in omics1_columns: # Green: mRNA
        new_img[:, x, y, 1] = X_img[:, x, y, 1]

    elif k in omics2_columns: # Green: lipid
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    elif k in clinicalomics1_columns: # Yellow (RG)
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 1] = X_img[:, x, y, 1]

    elif k in clinicalomics2_columns: # Pink (RB)
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    elif k in omics1omics2_columns: # Skyblue (GB)
        new_img[:, x, y, 1] = X_img[:, x, y, 1]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

    else: # RGB: virusmrnalipid
        new_img[:, x, y, 0] = X_img[:, x, y, 0]
        new_img[:, x, y, 1] = X_img[:, x, y, 1]
        new_img[:, x, y, 2] = X_img[:, x, y, 2]

X_test_img = new_img

## control image data
new_img = np.zeros_like(X_img)
new_img2 = np.ones_like(X_img)

for k in range(0, X_test.shape[1]-1):
    x = coords[k][0]
    y = coords[k][1]

    if k in clinical_columns: # Red: Virus
        new_img[:, x, y, 0] = new_img2[:, x, y, 0]

    elif k in omics1_columns: # Green: mRNA
        new_img[:, x, y, 1] = new_img2[:, x, y, 1]

    elif k in omics2_columns: # Blue: lipid
        new_img[:, x, y, 2] = new_img2[:, x, y, 2]

    elif k in clinicalomics1_columns: # Yellow (RG)
        new_img[:, x, y, 0] = new_img2[:, x, y, 0]
        new_img[:, x, y, 1] = new_img2[:, x, y, 1]

    elif k in clinicalomics2_columns: # Pink (RB)
        new_img[:, x, y, 0] = new_img2[:, x, y, 0]
        new_img[:, x, y, 2] = new_img2[:, x, y, 2]

    elif k in omics1omics2_columns: # Skyblue (GB)
        new_img[:, x, y, 1] = new_img2[:, x, y, 1]
        new_img[:, x, y, 2] = new_img2[:, x, y, 2]

    else: # White (RGB): virusmrnalipid
        new_img[:, x, y, 0] = new_img2[:, x, y, 0]
        new_img[:, x, y, 1] = new_img2[:, x, y, 1]
        new_img[:, x, y, 2] = new_img2[:, x, y, 2]

X_cont_img = new_img

# save X_train/test_img and y_train/test
np.save('outputs/X_train_img.npy', X_train_img)
np.save('outputs/X_test_img.npy', X_test_img)
y_train_sm.to_pickle('outputs/y_train.pkl')
y_test.to_pickle('outputs/y_test.pkl')

# View overall feature overlap
fdm = it.feature_density_matrix()
fdm[fdm == 0] = np.nan

plt.figure(figsize=(10, 7.5))
ax = sns.heatmap(fdm,
                 cmap = "viridis",
                 linewidths = 0.,
                 linecolor = "lightgrey",
                 square = True)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

for _, spine in ax.spines.items():
    spine.set_visible(True)
_ = plt.title("Genes per pixel")

plt.show()

## plot for each label
fix, axs = plt.subplots(ncols = 5,
                        nrows = num_classes, # 2
                        figsize = (40, 15))

for c in range(num_classes):
  indexes = np.where(y_train == c)[0]
  X_tmp = X_train_img[indexes]
  y_tmp = y_train.iloc[indexes]

  for i in range(5):
    img = X_tmp[i]
    label = y_tmp.iloc[i]
    print(f"{img.shape}, {img.min()}, {img.mean(axis=(0, 1))}, {img.max()}")
    axs[c, i].imshow(img)
    axs[c, i].set_title(f"sample {i}, label {label}")

fix.tight_layout()
plt.show()

## plot the control image
fix, axs = plt.subplots(ncols = 5,
                        nrows = num_classes,
                        figsize = (40, 15))

for c in range(num_classes):
  indexes = np.where(y_test == c)[0]
  X_tmp = X_cont_img[indexes]
  y_tmp = y_test.iloc[indexes]

  for i in range(5):
    img = X_tmp[i]
    label = y_tmp.iloc[i]
    print(f"{img.shape}, {img.min()}, {img.mean(axis=(0, 1))}, {img.max()}")
    axs[c, i].imshow(img)
    axs[c, i].set_title(f"sample {i}, label {label}")

fix.tight_layout()
plt.show()

# save images
import os
from PIL import Image

output_img_dir = "images"

for split, X, y in zip(['train', 'test'], [X_train_img, X_test_img], [y_train, y_test]):
  for c in range(num_classes):
    indexes = np.where(y == c)[0]
    X_tmp = X[indexes]
    y_tmp = y.iloc[indexes]

    output_img_class_dir = os.path.join(output_img_dir,
                                        split,
                                        str(c))

    os.makedirs(output_img_class_dir,
                exist_ok = True)

    print(f"saving {len(y_tmp)} {split} images for class {c} to {output_img_class_dir}")

    for i in range(len(y_tmp)):
      idx = indexes[i]
      img = X_tmp[i]
      label = y_tmp.iloc[i]

      img_path = os.path.join(output_img_class_dir,
                              f"idx{idx}_class{label}.png")

      Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
