# Set up environment variables
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Imports
import torch
import mmdet
import mmcv
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from mmdet.apis import show_result_pyplot
import os.path as osp
import seaborn as sns

# Modify the config
#cfg = Config.fromfile('/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py')
#cfg = Config.fromfile('/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py')
#cfg = Config.fromfile('/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py')
cfg = Config.fromfile('/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/aug_deformable_detr_r50_16x2_50e_coco.py')

cfg.dataset_type = 'COCODataset'
cfg.data.test.ann_file = '/home/habtamu/MMDetection/Dataset/validation_images/validation_annotations_modified.json'
cfg.data.test.img_prefix = '/home/habtamu/MMDetection/Dataset/validation_images/'
cfg.data.test.classes = ('lesion',)

cfg.data.train.ann_file = '/home/habtamu/MMDetection/Dataset/training_images/training_annotations_modified.json'
cfg.data.train.img_prefix = '/home/habtamu/MMDetection/Dataset/training_images/'
cfg.data.train.classes = ('lesion',)

cfg.data.val.ann_file = '/home/habtamu/MMDetection/Dataset/validation_images/validation_annotations_modified.json'
cfg.data.val.img_prefix = '/home/habtamu/MMDetection/Dataset/validation_images/'
cfg.data.val.classes = ('lesion',)

cfg.model.bbox_head.num_classes = 1
cfg.load_from = 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
#cfg.load_from = 'checkpoints/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth'
#cfg.load_from = 'checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
cfg.work_dir = '/home/habtamu/MMDetection/output/'

cfg.optimizer.lr = 0.0000125
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

#cfg.evaluation.metric = 'mAP'
cfg.runner.max_epochs = 28
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.device = 'cuda'
cfg.gpu_ids = range(1)

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

print(f'Config:\n{cfg.pretty_text}')

# Train a new detector
datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

# Save config to the file
# config_file = '/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco_default_28.py'
# config_file = '/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco_32.py'
# config_file = '/home/habtamu/MMDetection/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_24.py'
config_file = '/home/habtamuMMDetection/aug_deformable_detr_r50_16x2_50e_coco_28.py'
meta = dict()
meta['exp_name'] = osp.basename(config_file)
print(meta)
cfg.dump(osp.join("my_configs", meta['exp_name']))