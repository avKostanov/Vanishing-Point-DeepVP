import torch
from tqdm import tqdm
import config
from tools import *
import glob
import numpy as np

from model import DeepVP
from road_dataset import RoadTestDataset, create_test_meta
import re
import warnings
warnings.filterwarnings("ignore") 


def calculate(gt, predicted, path_img):
    
        with open(gt) as f:
            ground_truth = json.load(f)

        with open(predicted) as f2:
            predicted = json.load(f2)

        return calc_metrics(predicted, ground_truth, path_img)

if __name__ == '__main__':
    
    meta = create_test_meta(config.PATH_TO_TEST_DATASET)
    dataset = RoadTestDataset(meta)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = DeepVP()
    model.load_state_dict(torch.load(config.PATH_TO_WEIGHTS, map_location=config.DEVICE))

    with torch.no_grad():
        preds = []
        for batch in tqdm(test_loader):
            images = batch[0].to(config.DEVICE).unsqueeze(0)
            pred = model(images)
            preds.append(pred.squeeze(0).tolist())
            

    meta['predictions'] = preds
    to_save = meta[['filepath', 'predictions']]

    for idx in range(len(to_save)):
        to_save['filepath'].loc[idx] = os.path.basename(to_save['filepath'].loc[idx])

    to_save.set_index('filepath', inplace=True, drop=True)
    dict_to_save = to_save.T.to_dict('records')

    with open(config.PATH_TO_PREDICTIONS, 'w') as save:
        json.dump(dict_to_save[0], save)


