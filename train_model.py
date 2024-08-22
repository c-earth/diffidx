import torch
import time

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

model_dir = './models/'
raw_data_dir = './raw_data/'
raw_data_file = ''

