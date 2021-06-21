import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
import torchvision

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import glob
import os
import json
import random

import tensorboard
from torch.utils.tensorboard import SummaryWriter

from model import DeepVP
from road_dataset import RoadDataset, create_meta
from tools import *


import warnings
warnings.filterwarnings("ignore") 
from tools import *
import config


writer = SummaryWriter(config.PATH_TO_LOGS)
model = DeepVP()
criterion = config.LOSS_FUNCTION
optimizer = torch.optim.AdamW(model.parameters())

if __name__ == "__main__":
    for epoch in range(config.NUM_EPOCHS):
        generate_ds(config.PATH_TO_TEST_DATASET, config.PATH_TO_DATASET, 100, np.random.randint(0, 10000))
        meta = create_meta(config.PATH_TO_TEST_DATASET + '/', config.PATH_TO_TEST_MARKUP)
        
        valid_split = np.random.rand(len(meta)) < config.VALID_SIZE
        train = meta[valid_split]
        test = meta[~valid_split]

        dataset_train = RoadDataset(train)
        dataset_test = RoadDataset(test)

        training_loader = torch.utils.data.DataLoader(dataset_train, batch_size=12)
        validation_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1)

        epoch_loss = 0
        for batch in training_loader:
            images, coordinates = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            
            pred = model(images)
            loss = criterion(pred, coordinates)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.cpu().numpy() 
            
        writer.add_scalar('Train L1 Loss', epoch_loss/len(training_loader), epoch)
        torch.save(model.state_dict(), config.PATH_TO_WEIGHTS)
        with torch.no_grad():
            losses = []
            for batch in validation_loader:
                images, coordinates = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
                
                pred = model(images)
                loss = criterion(pred, coordinates)
                
                losses.append(loss)
        print(f'Epoch {epoch+0:03}: | Train Loss: {epoch_loss/len(validation_loader):.5f} | Valid Loss : {sum(losses)/len(losses):.5f}')
        writer.add_scalar('Valid L1 Loss', sum(losses)/len(losses), epoch)
        files = glob.glob(config.PATH_TO_TEST_DATASET + '/*')
        for f in files:
            os.remove(f)
