import torch

# training
DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 50
VALID_SIZE = 0.8
LOSS_FUNCTION = torch.nn.L1Loss()

# test: 
GENERATE_DATASET = True
COMPUTE_ERROR = True
DELETE_AFTER = True

# paths
PATH_TO_DATASET = '/home/alexey/vanishing_point/data'
PATH_TO_TEST_DATASET = '/home/alexey/vanishing_point/test'
PATH_TO_TEST_MARKUP = '/home/alexey/vanishing_point/test/markup.json'
PATH_TO_WEIGHTS = '/home/alexey/vanishing_point/deepvp.pth'
PATH_TO_LOGS = '/home/alexey/vanishing_point/logs'
PATH_TO_PREDICTIONS = '/home/alexey/vanishing_point/predicted.json'
