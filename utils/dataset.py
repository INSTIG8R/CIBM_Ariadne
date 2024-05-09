import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import cv2
from scipy.ndimage.interpolation import zoom

def clean_lists(file_list, other_list, folder_path):
  """
  This function takes two lists (file_list and other_list) and a folder path. 
  It removes entries from both lists where the corresponding filename in file_list 
  doesn't exist in the folder.
  """
  cleaned_indices = [i for i, filename in enumerate(file_list) if os.path.isfile(os.path.join(folder_path, filename))]
  return [file_list[i] for i in cleaned_indices], [other_list[i] for i in cleaned_indices]

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        # self.sample_list = open('/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/train.txt').readlines()

        self.output_size = image_size

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        image_list = list(self.data['Image'])
        caption_list = list(self.data['Description'])

        # folder_path = '/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/train_npz'

        folder_path = '/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Test Set/synapse_npz_from_h5' #For test

        self.image_list, self.caption_list = clean_lists(image_list, caption_list, folder_path)



        if mode == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        else:
            pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        # slice_name = self.sample_list[idx].strip('\n')
        # data_path = '/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/train_npz/'+slice_name+'.npz'
        # data = np.load(data_path)
        # image, label = data['image'], data['label']

        # trans = self.transform(self.image_size)

        image_list = self.image_list
        # data_path = os.path.join(self.root_path, image_list[idx])

        npz_file = os.path.join(self.root_path,'synapse_npz_from_h5',image_list[idx])
        npz_data = np.load(npz_file)
        
        image = npz_data['image']
        gt = npz_data['label']

        unique_values = np.unique(gt)
        num_classes = len(unique_values)

        caption = self.caption_list[idx]

        

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=50, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'token':token, 'mask':mask}
        # data = trans(data)

        # image_np = cv2.imread(gt)
        # self.image_np.append(image_np)
        # flat_data = np.concatenate(self.image_np)
        # unique_elements = np.unique(flat_data)

        # image_np_unique = np.unique(image_np)

        token,mask = data['token'],data['mask']
        # gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        gt_reshaped = np.zeros((9, gt.shape[0], gt.shape[1]), dtype=gt.dtype)

        for i in range(8):
            gt_reshaped[i] = (gt == i).astype(gt.dtype)

        gt = torch.from_numpy(gt_reshaped.astype(np.int32))

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # gt = torch.from_numpy(gt.astype(np.float32))

        return ([image, text], gt)

    # def transform(self,image_size=[224,224]):

    #     if self.mode == 'train':  # for training mode
    #         trans = Compose([
    #             LoadImaged(["image","gt"], reader='PILReader'),
    #             EnsureChannelFirstd(["image","gt"]),
    #             RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
    #             Resized(["image"],spatial_size=image_size,mode='bicubic'),
    #             Resized(["gt"],spatial_size=image_size,mode='nearest'),
    #             NormalizeIntensityd(['image'], channel_wise=True),
    #             ToTensord(["image","gt","token","mask"]),
    #         ])
        
    #     else:  # for valid and test mode: remove random zoom
    #         trans = Compose([
    #             LoadImaged(["image","gt"], reader='PILReader'),
    #             EnsureChannelFirstd(["image","gt"]),
    #             Resized(["image"],spatial_size=image_size,mode='bicubic'),
    #             Resized(["gt"],spatial_size=image_size,mode='nearest'),
    #             NormalizeIntensityd(['image'], channel_wise=True),
    #             ToTensord(["image","gt","token","mask"]),

    #         ])

    #     return trans


