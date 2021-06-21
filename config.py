import torch

# training
DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# paths
PATH_TO_TEST_DATASET = '/home/alexey/vanishing_point/test'
PATH_TO_WEIGHTS = '/home/alexey/vanishing_point/deepvp.pth'
PATH_TO_PREDICTIONS = '/home/alexey/vanishing_point/predicted.json'
