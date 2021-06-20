import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import json
import pandas as pd

class RoadDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image = Image.open(self.df['filepath'].iloc[i])

        image = ToTensor()(image)

        coordinates = torch.Tensor(self.df['coordinates'].loc[i])
        return image, coordinates

def create_meta(path_to_dataset, path_to_markup):
    
    with open(path_to_markup) as f:
        data = json.load(f)

    files = []
    labels = []
    for file, point in data.items():
        files.append(path_to_dataset + file)
        labels.append(point)
    df = pd.DataFrame()
    df['filepath'] = files
    df['coordinates'] = labels

    return df
