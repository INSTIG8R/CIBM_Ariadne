import json
import os
import torch
import pandas as pd
from monai.transforms import (Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

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

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        image_list = list(self.data['Image'])
        caption_list = list(self.data['Description'])

        folder_path = '/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/MosMedData+/Ariadne/Train Set/Images'

        # folder_path = '/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/MosMedData+/Ariadne/Test Set/Images'  #For test

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

        trans = self.transform(self.image_size)

        image_list = self.image_list
        
        image = os.path.join(self.root_path,'Images',image_list[idx])
        gt = os.path.join(self.root_path,'Ground-truths', image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=50, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans

