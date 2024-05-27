import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models.ours import ours
from dataset.npy_datasets import NPY_datasets
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"
from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

config = setting_config
print('#----------GPU init----------#')
set_seed(config.seed)
gpu_ids = [0]# [0, 1, 2, 3]
torch.cuda.empty_cache()
val_dataset = NPY_datasets(config.data_path, config, train=False)
val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)

print('#----------Prepareing Models----------#')
model_cfg = config.model_config
model = ours(num_classes=model_cfg['num_classes'],
             input_channels=model_cfg['input_channels']
             )

model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
print('#----------Prepareing loss, opt, sch and amp----------#')
criterion = config.criterion
optimizer = get_optimizer(config, model)
scheduler = get_scheduler(config, optimizer)
scaler = GradScaler()
dir='checkpoints/'+config.datasets+'/'
outputs_dir=config.datasets+'/'
if os.path.exists(os.path.join(dir, 'best.pth')):
    print('#----------Testing----------#')
    best_weight = torch.load(dir + 'best.pth', map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    loss = test_one_epoch(
        val_loader,
        outputs_dir,
        model,
        criterion,
        config,
    )