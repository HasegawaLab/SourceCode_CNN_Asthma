import timm
from timm.scheduler import create_scheduler_v2
timm.list_models(pretrained = True)

import torchsummary
import glob
import os
from torchvision.datasets import ImageFolder

from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

## Import image data
input_img_dir = "images"

train_dataset = ImageFolder(os.path.join(input_img_dir, "train"))
val_dataset = ImageFolder(os.path.join(input_img_dir, "test")) 
test_dataset = ImageFolder(os.path.join(input_img_dir, "test"))

## Show image examples
os.environ["WANDB_START_METHOD"] = "thread"
plt.rcParams["savefig.bbox"] = 'tight' 

def show(imgs, labels):
    if not isinstance(imgs, list):
        imgs = [imgs] 

    fig, axs = plt.subplots(ncols = len(imgs),
                            squeeze = False,
                            figsize = (30, 30))

    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set_title(str(labels[i]))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

imgs = [train_dataset[i][0] for i in range(5)]
labels = [train_dataset[i][1] for i in range(5)]
show(imgs, labels)

## Init weights and biases
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb
import timm.utils.random

cudnn.benchmark = True 
timm.utils.random.random_seed(seed = 42)

config = {'lr': 1e-5,
          'epochs': 500,
          'batch_size': 32}

mixup_args = {'mixup_alpha': 1.,
              'cutmix_alpha': 0.,
              'cutmix_minmax': None,
              'prob': .0,  # off
              'switch_prob': 0.0,
              'mode': 'batch',
              'label_smoothing': 0.1,
              'num_classes': 2}

config.update(mixup_args)

wandb.init(project = 'YOURPROJECT',
           entity = 'YOURUERNAME',
           config = config)

## Data transformation for analysis

from timm.data.mixup import Mixup

if mixup_args['prob'] > 0.0 and mixup_args['mixup_alpha'] > 0.0:
  mixup_fn = Mixup(**mixup_args) # **, キーワード引数を指定する
else:
  mixup_fn = None

mixup_fn

from albumentations.augmentations.transforms import MultiplicativeNoise
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

### {}, make dictionary
image_datasets = {x: datasets.ImageFolder(os.path.join(input_img_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

image_datasets['val'] = image_datasets['test']  # re-use test set

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = 32,
                                              shuffle = True,
                                              num_workers = 1,
                                              drop_last = (x == 'train'))
              for x in ['train', 'val', 'test']}

### drop_last for mixup during train
dataset_sizes = {x: len(image_datasets[x])
                for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Show tensor as image, asthma = 0
### function to show tensor as image
def imshow(inp,
           title = None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize = (30, 30))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

### Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

target_class = 0
idx = classes == target_class

### Make a grid from batch
out = torchvision.utils.make_grid(inputs[idx])

imshow(out,
       title = [class_names[x] for x in classes[idx]])

## Show tensor as image, asthma = 1
target_class = 1
idx = classes == target_class

### Make a grid from batch
out = torchvision.utils.make_grid(inputs[idx])

imshow(out, title=[class_names[x] for x in classes[idx]])

## Evaluate results
print(classes)

### Get a batch of training data
if mixup_fn is not None:
  mix_inputs, mix_classes = mixup_fn(inputs, classes)

### Make a grid from batch
  out = torchvision.utils.make_grid(mix_inputs)

  print(mix_classes)

  imshow(out,
         title = [class_names[x] for x in mix_classes.argmax(1)])

from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate(net,
             dataloader):
  predicted_scores = np.empty(0)
  true_labels = np.empty(0)
  predicted_labels = np.empty(0)

  with torch.no_grad():
      net.eval() 
      for i, data in enumerate(dataloader):
          inputs, labels = data
          pred = torch.max(net(inputs.cuda()),
                           1)[1].cpu().detach().numpy()
          predicted_scores = np.append(predicted_scores,
                                       pred)
          true_labels = np.append(true_labels,
                                  labels.cpu().detach().numpy())
          predicted_labels = np.append(predicted_labels, (pred > 0.5).astype(int))

  assert predicted_scores.shape == true_labels.shape
  assert true_labels.shape == predicted_labels.shape

  metrics = {
      "accuracy": accuracy_score(true_labels,
                                 predicted_scores),
      "auroc": roc_auc_score(true_labels,
                             predicted_scores),
      "pred_pos_rate": np.mean(predicted_labels),
      "true_pos_rate": np.mean(true_labels),
      "n_samples": len(true_labels)
  }


  return predicted_scores, true_labels, metrics

## Define {train_model}
def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs = 100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auroc = 0.0

    ema_model = None 
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test', 'val']:
          if ema_model is not None:
            _, _, metrics = evaluate(ema_model.module,
                                     dataloaders[phase])
          else:
            _, _, metrics = evaluate(model,
                                     dataloaders[phase])

          for k, v in metrics.items():
            wandb.log({f'{phase}_{k}': v},
                      step = epoch)

          if phase == 'val':
            val_auroc = metrics['auroc']
            print("val auroc", val_auroc)

            if best_auroc < val_auroc:
                best_auroc = val_auroc

                if ema_model is not None:
                  best_model_wts = copy.deepcopy(ema_model.module.state_dict())
                else:
                  best_model_wts = copy.deepcopy(model.state_dict())

                print(f"best score {best_auroc}")
                print("saving model weights")
                torch.save(model.state_dict(), "best_model.pth")

        ### Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  ### Set model to training mode
            else:
                model.eval()   ### Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            ### Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if mixup_fn is not None:
                  inputs, labels = mixup_fn(inputs,
                                            labels)

                ### zero the parameter gradients
                optimizer.zero_grad()

                ### forward
                ### track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast("cuda"):
                      outputs = model(inputs)
                      _, preds = torch.max(outputs,
                                           1)
                      loss = criterion(outputs,
                                       labels)

                    ### backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if ema_model is not None:
                           ema_model.update(model)

                ### statistics
                running_loss += loss.item() * inputs.size(0)
                if mixup_fn is not None:
                  labels_max = labels.argmax(1)
                  running_corrects += torch.sum(preds == labels_max)
                else:
                  running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            wandb.log({f'{phase}_loss': epoch_loss,
                       'lr': optimizer.param_groups[0]["lr"]},
                      step = epoch)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test auroc: {best_auroc:4f}')

    torch.save(model.state_dict(),
               "last_model.pth")

    artifact = wandb.Artifact('last_model',
                              type = 'model')
    artifact.add_file('last_model.pth')
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact('best_model',
                              type = 'model')
    artifact.add_file('best_model.pth')
    wandb.log_artifact(artifact)

    import datetime
    datetime_str = datetime.datetime.fromtimestamp(wandb.run.start_time).strftime('%Y-%m-%d-%H-%M-%S')

    drive_save_dir = f"/content/drive/MyDrive/00_Colab/Tao/models/{datetime_str}_{wandb.run.name}"
    os.makedirs(drive_save_dir,
                exist_ok = True)

    import shutil
    shutil.copy('last_model.pth',
                os.path.join(drive_save_dir,
                             'last_model.pth'))
    shutil.copy('best_model.pth',
                os.path.join(drive_save_dir,
                             'best_model.pth'))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

## Define {creat_resnet}
def create_resnet(dropout_rate = 0.5):
  model_ft = models.resnet50(pretrained = True)
  num_ftrs = model_ft.fc.in_features 
  
  if dropout_rate > 0.0:
    model_ft.fc = nn.Sequential(nn.Dropout(dropout_rate),
                                nn.Linear(num_ftrs, 2))
  else:
    model_ft.fc = nn.Linear(num_ftrs, 2)

  return model_ft

## Define {create_squeezenet}

def create_squeezenet():
  model_ft = models.squeezenet1_1(pretrained = True)

  model_ft.classifier[1] = nn.Conv2d(512,
                                     2,
                                     kernel_size = (1, 1),
                                     stride = (1, 1))

  return model_ft

## Define {create_model_and_optim}

def create_model_and_optim(model_name = 'resnet50',
                           lr = 1e-4,
                           drop_rate = 0.5,
                           num_epochs = 1000,
                           weight_decay = 0.1,
                           layer_decay = 0.75):
  if model_name == 'torchvision_resnet50':
    model_ft = create_resnet()

  elif model_name == 'squeezenet':
    model_ft = create_squeezenet()

  else:
    import timm
    model_ft = timm.create_model(model_name,
                                 pretrained = True,
                                 num_classes = 2,
                                 drop_rate = drop_rate)

  model_ft = model_ft.to(device)

  from sklearn.utils.class_weight import compute_class_weight

  train_y = np.array(image_datasets['train'].targets)
  class_weights = compute_class_weight(class_weight = 'balanced',
                                       classes = np.unique(train_y),
                                       y = train_y)

  class_weights = torch.tensor(class_weights,
                               dtype = torch.float).to(device)
  print("class_weights",
        class_weights)

  from timm.loss import SoftTargetCrossEntropy, BinaryCrossEntropy

  criterion = BinaryCrossEntropy(smoothing = 0.1,
                                 reduction = 'mean',
                                 pos_weight = class_weights)
  from timm.optim import create_optimizer_v2, optimizer_kwargs
  optimizer_ft = create_optimizer_v2(model_ft.parameters(),
                                     opt = 'AdamW',
                                     lr = lr,
                                     weight_decay = weight_decay,
                                     layer_decay = layer_decay)

  from timm.scheduler import create_scheduler_v2
  scheduler, num_epochs = create_scheduler_v2(optimizer_ft,
                                              warmup_lr = 1e-9,
                                              warmup_epochs = 50,
                                              num_epochs = num_epochs)

  return model_ft, optimizer_ft, scheduler, criterion

## Define {objective}
def objective(trial):
  params = {
      'lr': trial.suggest_categorical('lr', [3e-6]),
      'epochs': trial.suggest_categorical('epochs', [1500]),
      'encoder': trial.suggest_categorical('encoder', ['squeezenet']),
       #, 'tv_resnet152', 'swin_large_patch4_window7_224_in22k'])
      'drop_rate': trial.suggest_categorical('drop_rate', [None]),
      'layer_decay': trial.suggest_categorical('layer_decay', [None]),
      'weight_decay': trial.suggest_categorical('weight_decay', [0.01]),
  }

  config = dict(trial.params)
  config["trial.number"] = trial.number
  print(config)

  wandb.init(
      project = "MARC35",
      entity = 'shibataryohei',
      # entity = "tadook",
      config = config,
      group = "0419_2_squeezenet_save",
      reinit = True
  )

  timm.utils.random.random_seed(seed = 42)

  model_ft, optimizer_ft, scheduler, criterion = create_model_and_optim(model_name = params['encoder'],
                                                                        num_epochs = params['epochs'],
                                                                        lr = params['lr'],
                                                                        drop_rate = params['drop_rate'],
                                                                        layer_decay = params['layer_decay'],
                                                                        weight_decay = params['weight_decay'])

  torchsummary.summary(model_ft, (3, 227, 227),
                       device = "cuda")

  model_ft = train_model(model_ft,
                         criterion,
                         optimizer_ft,
                         scheduler,
                         num_epochs = params['epochs'])

  test_predicted, test_true, test_metrics = evaluate(model_ft,
                                                     dataloaders['test'])
  print(test_metrics)
  wandb.log({"final_test_" + k: v for k, v in test_metrics.items()})

  val_predicted, val_true, val_metrics = evaluate(model_ft,
                                                  dataloaders['val'])
  print(val_metrics)
  wandb.log({"final_val_" + k: v for k, v in val_metrics.items()})

  primary_metric = val_metrics['auroc']
  wandb.run.summary["final val auroc"] = primary_metric
  wandb.run.summary["state"] = "completed"
  wandb.finish(quiet=True)

  return primary_metric

import optuna
study = optuna.create_study(direction = "maximize",
                            pruner = optuna.pruners.MedianPruner())

study.optimize(objective,
               n_trials = 1,
               timeout = 1200000)

loaded_model, _, _, _ = create_model_and_optim("squeezenet")

loaded_model.load_state_dict(torch.load('best_model.pth'))
loaded_model.load_state_dict(torch.load('last_model.pth'))
