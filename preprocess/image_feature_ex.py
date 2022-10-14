import json
import numpy as np
import random
import os, errno
import sys
from os import listdir
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch
from model.encoder import Image_Encoder
import pickle

def main():    
    
    anno_path="/HadiSheikhi/fsvqa_data_train/annotations.pickle"
    
    with open(anno_path, 'rb') as fid:
        train_anno = pickle.load(fid)

    # with open('data/coco_ids/annotations/instances_train2014.json') as f:
    #     data = json.load(f)

    image_ids=[]
    if 'annotations' in train_anno:
                for ann in train_anno['annotations']:
                    image_ids.append(ann['image_id'])

    image_trans=[]
    coco_image_features=[]
    folder_dir = 'data/resize_image/train2014'

    transform = transforms.Compose([
            transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
        ])

    zeros=""

    img_encoder = Image_Encoder(1024)

    for start_idx in tqdm(range(0, len(image_ids), 32)):
        ids=image_ids[start_idx: start_idx + 32]

        for image_id in ids:
            
            if int(image_id)<10:
                zeros="00000000000"
            elif int(image_id)<100:
                zeros="0000000000"  
            elif int(image_id)<1000:
                zeros="000000000"  
            elif int(image_id)<10000:
                zeros="00000000"     
            elif int(image_id)<100000:
                zeros="0000000"      
            elif int(image_id)<1000000:
                zeros="000000"      
            img = np.array(Image.open(folder_dir+'COCO_train2014_'+zeros+str(image_id)+'.jpg').convert('RGB'))
            img = transform(img)
            image_trans.append(img)

        image_trans = torch.stack(image_trans, dim = 0)
        coco_image_features = img_encoder(image_trans)
        image_trans = []
       
