import torch
from tqdm import tqdm
import config
from tools import *
import glob
import numpy as np

from model import DeepVP
from road_dataset import RoadDataset, create_meta
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
    if config.GENERATE_DATASET:
        if not os.path.exists(config.PATH_TO_TEST_DATASET):
            os.mkdir(config.PATH_TO_TEST_DATASET)
        generate_ds(config.PATH_TO_TEST_DATASET, config.PATH_TO_DATASET, num=50, seed=np.random.randint(0, 10000))
    
    meta = create_meta(config.PATH_TO_TEST_DATASET + '/', config.PATH_TO_TEST_MARKUP)

    dataset = RoadDataset(meta)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = DeepVP()
    model.load_state_dict(torch.load(config.PATH_TO_WEIGHTS, map_location=config.DEVICE))

    with torch.no_grad():
        preds = []
        for batch in tqdm(test_loader):
            images, coordinates = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            
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

        
    print(calculate(
        gt=config.PATH_TO_TEST_MARKUP , 
        predicted=config.PATH_TO_PREDICTIONS, 
        path_img=config.PATH_TO_TEST_DATASET))

    if config.DELETE_AFTER:
        files = glob.glob(config.PATH_TO_TEST_DATASET + '/*')
        for f in files:
            os.remove(f)