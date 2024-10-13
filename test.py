import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import seaborn as sns
import albumentations  as albu
from albumentations.pytorch import ToTensorV2

import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import models
from torch.autograd import Function
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

input_dir = "data"
# input_dir_Test  = os.path.join(input_dir, 'NEU_Seg/images/train_images')
# filelist_Test = os.listdir(input_dir_Test)
# test_df_path = os.path.join(input_dir, 'NEU_trainval_test.csv')

input_dir_Test  = os.path.join(input_dir, 'NEU_Seg/images/test')
filelist_Test = os.listdir(input_dir_Test) #['000001.jpg', '000002.jpg',...]
test_df_path = os.path.join(input_dir, 'NEU-Test.csv')

test_df = pd.read_csv(test_df_path)
# test_df.head()

def make_df(tdf):
    tdf = tdf.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    tdf['defects'] = tdf.count(axis=1)
    
    return tdf

test_data = make_df(test_df)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
def get_augmentation(mean, std, phase):
    
    if phase == 'train':
        transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # albu.RandomCrop(256, 512, p=1),
            albu.Resize(256, 256, interpolation=cv2.INTER_NEAREST, p=1),
            albu.Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    else:
        transform = [
            # albu.RandomCrop(256, 256, p=1),
            albu.Resize(256, 256, interpolation=cv2.INTER_NEAREST, p=1),
            albu.Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    
    return albu.Compose(transform)
def make_mask(index, df):
    filename = df.iloc[index].name
    labels = df.iloc[index, :3]
    masks = np.zeros((200, 200, 3), dtype=np.float32)
    for idx, label in enumerate(labels):
        if label is not np.nan:
            mask = np.zeros((200*200), dtype=np.uint8)
            pixels = label.split(' ')
            pixels = [pixels[i:i+2] for i in range(0, len(pixels), 2)]
            for pixel in pixels:
                pos, le = pixel
                pos, le = int(pos), int(le)
                mask[pos-1:pos+le-1] = 1
            masks[:,:,idx] = mask.reshape(200, 200, order = 'F')
    return filename, masks


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, df, phase):
        self.df = df
        self.input_dir = input_dir
        # self.fnames = df['ImageId'].unique().tolist()
        self.transforms = get_augmentation(mean, std, phase) 
        self.phase = phase
    def __getitem__(self, idx):

        # fname = self.fnames[idx]
        fname, mask = make_mask(idx, self.df)
        image = cv2.imread(os.path.join(self.input_dir, fname))
        augmented = self.transforms(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        mask = mask.permute(2, 0, 1)
        return image, mask, fname
    def __len__(self):
        return len(self.df)
        
def decode_segmap(image, nc=4):
  
  label_colors = np.array([(0,0,0), (0,128,0), (0, 128, 128), (192, 128, 128)])

            # 0=background, # 1=Inclusions, 2=Patches, 3=Scratches
            #[(0,0,0), (128,0,0), (0,128,0), (128,128,0)] the usual
            #Sctrach = (128,128,0)
            #Patches = (0,128,0)
            # Inclusion = (128,0,0)
            #                   (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            #    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            #    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

# from models.FPN import model as m #import all the required models accordingly ({ENEt, DANet, ENCNet, and etc})
# from models.proposed_models.segformer import*
# from models.proposed_models.Transformer_based import*
# from models.proposed_models.CNN_based import*
from models.UNet import*
batch_size = 1
test_dataset = TestDataset(input_dir_Test, test_data, phase = 'test')
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# net = m # Use this for CNN-based models
#model = net
# net = Transformer_based('ResT-S') #Proposed hybrid netwrok of trasnformer-encoder (ResT) and CNN-decoder (UperNet)
# net = CNN_based('ConvNeXt-S') #CNN-based hybrid Networks
# net.init_pretrained('.../pretrained_weights/cpt/rest_small.pth')
net = UNet()

model = net

def count_parameters(x):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))

# #Model inference time, FPS
# test_model = model
# test_model.eval()
# device = torch.device("cuda")
# test_model.to(device)
# dummy_input = torch.randn(1, 3,256,256, dtype=torch.float).to(device)
# # INIT LOGGERS
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 1000
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#     _ = test_model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter.record()
#         _ = test_model(dummy_input)
#         ender.record()
#         # WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)
#         timings[rep] = curr_time
# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print('Average time per image in ms: ', mean_syn)
# print('Standard deviation of time:', std_syn)
# print('Frames per second (FPS): ', (1/(mean_syn/1000)))
# Average time per image in ms:  22.16038399505615
# Standard deviation of time: 0.24000627057309856
# Frames per second (FPS):  45.125571841313494
# Best result found at epoch of:  144

checkpoint_0 = torch.load("./model_weights/Checkpoints/NEU_ResT_S_UperHead.pth")

model.load_state_dict(checkpoint_0['state_dict']) 
print('Best result found at epoch of: ',checkpoint_0['epoch'])
def dice_coeff(pred, mask):
    with torch.no_grad():
        batch_size = len(pred)
        pred = pred.view(batch_size, -1) # Flatten
        mask = mask.view(batch_size, -1)  # Flatten
        pred = (pred>0.5).float()
        mask = (mask>0.5).float()
        smooth = 0.0001
        intersection = (pred * mask).sum()
        dice = (2. * intersection +smooth ) / (pred.sum() + mask.sum() + smooth) 
        # intersection = ((pred + mask) == 0).sum()
        # dice_neg = (2. * intersection ) / ((pred == 0).sum() + (mask == 0).sum() + smooth)
        # dice = (dice_pos + dice_neg) / 2.0
        return dice #.item()

def IOU(pred, mask):
    with torch.no_grad():
        batch_size = len(pred)
        pred = pred.view(batch_size, -1) # Flatten
        mask = mask.view(batch_size, -1)  # Flatten
        pred = (pred>0.5).float()
        mask = (mask>0.5).float()
        smooth = 0.0001
        intersection = (pred * mask).sum() + smooth
        union = (pred.sum() + mask.sum()) - intersection + smooth
        iou = intersection / union 
        # intersection = ((pred + mask) == 0).sum()
        # dice_neg = (2. * intersection ) / ((pred == 0).sum() + (mask == 0).sum() + smooth)
        # dice = (dice_pos + dice_neg) / 2.0
        return iou #.item()


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric_dice (probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice # dice_neg, dice_pos, num_neg, num_pos
def class_dice(pred, mask):
    with torch.no_grad():
        batch_size = len(pred)
        # pred = pred.view(batch_size, -1) # Flatten
        # mask = mask.view(batch_size, -1)  # Flatten
        pred = (pred>0.5) 
        mask = (mask>0.5) 
        smooth = 0.0001
        
        intersection = (pred * mask).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth) 
        # intersection = ((pred + mask) == 0).sum()
        # dice_neg = (2. * intersection ) / ((pred == 0).sum() + (mask == 0).sum() + smooth)
        # dice = (dice_pos + dice_neg) / 2.0
        return dice #.item()


def class_iou(pred, mask):
    with torch.no_grad():
        batch_size = len(pred)
        # pred = pred.view(batch_size, -1) # Flatten
        # mask = mask.view(batch_size, -1)  # Flatten
        pred = (pred>0.5) 
        mask = (mask>0.5) 
        smooth = 0.0001

        intersection = (pred * mask).sum()
        union = ((pred.sum() + mask.sum()) - intersection)
        iou = (intersection + smooth) / (union + smooth) 
        
        return iou #.item()
print('state_dict loaded') 

# Testing the model on test set
def model_test(model, dataloader):
    
    # GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available device：", device)

    # model to device
    model = model.to(device)
    model.eval()

     
    torch.backends.cudnn.benchmark = True
    
    num_test_imgs = len(test_dataloader.dataset)
    batch_size = test_dataloader.batch_size
    
    epoch_acc = 0.0
    class_1_dice = 0.0
    class_2_dice = 0.0
    class_3_dice = 0.0

    epoch_iou = 0.0
    class_1_iou = 0.0
    class_2_iou = 0.0
    class_3_iou = 0.0
    # class_4_dice = 0.0
    # dice_mean = 0.0
    class_1_num = 0
    class_2_num = 0
    class_3_num = 0

    class_1_iou_all =0.0
    class_2_iou_all = 0.0
    class_3_iou_all = 0.0
    with torch.no_grad():
        for img, mask, fname in tqdm(test_dataloader):

        # for img, mask, fname in test_dataloader:
            # model.eval()
            # data to device
            img = img.to(device)
            mask = mask.to(device) #torch.Size([1, 3, 256, 256])
            output = model(img) #torch.Size([1, 3, 256, 256])
            prob = torch.sigmoid(output)
            prob = prob.to('cpu').detach()
            mask = mask.to('cpu').detach()

            # prob = torch.sigmoid(output)
        
            # Calculating class dice

            img = img.to('cpu').detach().numpy().copy()[0,0] #(256, 256)
            msk_1 = mask.numpy().copy()[0,0] #(256, 256)
            msk_2 = mask.numpy().copy()[0,1]
            msk_3 = mask.numpy().copy()[0,2]
        
            prob_1 = prob.numpy().copy()[0,0]
            prob_2 = prob.numpy().copy()[0,1]
            prob_3 = prob.numpy().copy()[0,2]
       
                  
            epoch_acc += dice_coeff(prob, mask)
            class_1_dice += class_dice(prob_1, msk_1)
            class_2_dice += class_dice(prob_2, msk_2)
            class_3_dice += class_dice(prob_3, msk_3)
            
            # Calculating the iou and mean iou 
            epoch_iou += IOU(prob, mask)
            class_1_iou = class_iou(prob_1, msk_1)
            class_2_iou = class_iou(prob_2, msk_2)
            class_3_iou = class_iou(prob_3, msk_3)
            if class_1_iou == 1.0:
                class_1_num+=1
            if  class_2_iou == 1.0:
                class_2_num+=1
            if  class_3_iou == 1.0:
                class_3_num+=1
            # else :
            #     class_3_num+=0
            #     class_2_num+=0
            #     class_1_num+=0
                
            class_1_iou_all += class_1_iou
            class_2_iou_all += class_2_iou
            class_3_iou_all += class_3_iou
            # breakpoint()
    print(class_1_num,class_2_num,class_3_num)
    cl1_dice = class_1_dice / num_test_imgs * batch_size
    cl2_dice = class_2_dice / num_test_imgs * batch_size
    cl3_dice = class_3_dice / num_test_imgs * batch_size
    mDice = (cl1_dice + cl2_dice + cl3_dice)/3

    cl1_mIoU = (class_1_iou_all-class_1_num) / (num_test_imgs * batch_size-class_1_num)
    cl2_mIoU = (class_2_iou_all-class_2_num) / (num_test_imgs * batch_size-class_2_num)
    cl3_mIoU = (class_3_iou_all-class_3_num) / (num_test_imgs * batch_size-class_3_num)
    mIoU = (cl1_mIoU + cl2_mIoU + cl3_mIoU)/3
    
    return cl1_dice, cl2_dice, cl3_dice, mDice, cl1_mIoU, cl2_mIoU, cl3_mIoU, mIoU
cl1_dice, cl2_dice, cl3_dice, mDice, cl1_mIoU, cl2_mIoU, cl3_mIoU, mIoU = model_test(model, test_dataloader)
# import module
from tabulate import tabulate
 
# assign data
Results = [
    ["Score", cl1_dice*100, cl2_dice*100, cl3_dice*100, mDice*100, cl1_mIoU*100, cl2_mIoU*100, cl3_mIoU*100, mIoU*100]
]
 
# create header
head = ["Metrics", "cl1_dice", "cl2_dice", "cl3_dice", "mDice", "cl1_mIoU", "cl2_mIoU", "cl3_mIoU", "mIoU"]
 
# display table
print(tabulate(Results, headers=head, tablefmt="fancy_grid"))
