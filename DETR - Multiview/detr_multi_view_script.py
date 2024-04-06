# Set up environment variable
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
print(torch.__version__, torch.cuda.is_available())

import sys, distutils.core
dist = distutils.core.run_setup("./detectron2/setup.py")
sys.path.insert(0, os.path.abspath('./detectron2'))

# Import some common libraries
import numpy as np
import cv2
import random
import pandas as pd

# Import omidb
import omidb

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, pairwise_iou

# Include the directory to the Python path
# Otherwise, it will give the following error: No module named d2 
sys.path.append('/home/habtamu/DETR_Multi_View/detr')

# Register Data
def get_omidb_dicts(img_dir, csv_dir):
    df = pd.read_csv(csv_dir)
    
    dataset_dicts = []
    for idx, row in df.iterrows():
        record = {}
        filename = os.path.join(img_dir, row["filename"])

        record["file_name"] = filename
        record["image_id"] = idx
        
        if row['side'] == 'R':
            # Bounding box breast area         
            bbox = row["bbox"][12:-1]
            coords1 = bbox.split(',')
            r= np.array([0,0,0,0])
            indx1 = 0
            for c in coords1:
                aux = c.split('=')
                r[indx1]=(int(aux[1]))
                indx1 +=1

            # we can get width and heigth from bbox
            record["height"] = r[3]-r[1]
            record["width"] = r[2]-r[0]
            
            # Bounding box roi  
            bbox_roi = row["bbox_roi"][12:-1]
            coords2 = bbox_roi.split(',')
            s= np.array([0,0,0,0])
            indx2 = 0
            for c in coords2:
                aux = c.split('=')
                s[indx2]=(int(aux[1]))
                indx2 +=1
            bbox_roi = omidb.mark.BoundingBox(s[0]-r[0],s[1]-r[1],s[2]-r[0],s[3]-r[1])
            
        else:
            im = cv2.imread(filename)
            h,w,_ = im.shape
            record["height"] = h
            record["width"] = w

            # Bounding box roi  
            bbox_roi = row["transformed_bbox_roi"][12:-1]
            coords2 = bbox_roi.split(',')
            s= np.array([0,0,0,0])
            indx2 = 0
            for c in coords2:
                aux = c.split('=')
                float_value = round(float(aux[1]), 0)
                s[indx2]=(int(float_value))
                indx2 +=1
            bbox_roi = omidb.mark.BoundingBox(s[0],s[1],s[2],s[3])

        px = [bbox_roi.x1, bbox_roi.x2, bbox_roi.x2, bbox_roi.x1]
        py = [bbox_roi.y1, bbox_roi.y1, bbox_roi.y2, bbox_roi.y2]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        objs = []
        obj =  {
                "bbox": [bbox_roi.x1 , bbox_roi.y1, bbox_roi.x2, bbox_roi.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("omidb_train", lambda: get_omidb_dicts("/home/habtamu/Mammogram_Registration_Four_Resolution/stacked_without_difference_image","/home/habtamu/Mammogram_Registration_Four_Resolution/transformed_lesion_training_set.csv"))
MetadataCatalog.get("omidb_train").set(thing_classes=["lesion"])

DatasetCatalog.register("omidb_val", lambda: get_omidb_dicts("/home/habtamu/Mammogram_Registration_Four_Resolution/stacked_without_difference_image","/home/habtamu/Mammogram_Registration_Four_Resolution/transformed_lesion_validation_set.csv"))
MetadataCatalog.get("omidb_val").set(thing_classes=["lesion"])

# Training and val set dictionaries
train_metadata = MetadataCatalog.get("omidb_train")
train_dictionary = DatasetCatalog.get("omidb_train")

val_metadata = MetadataCatalog.get("omidb_val")
val_dictionary = DatasetCatalog.get("omidb_val")

# Set up the configuration for the omidb dataset
from d2.detr import add_detr_config
cfg = get_cfg()
add_detr_config(cfg)
cfg.merge_from_file("detr/d2/configs/detr_256_6_6_torchvision.yaml")
cfg.DATASETS.TRAIN = ("omidb_train",)
cfg.DATASETS.TEST = ("omidb_val",)
cfg.OUTPUT_DIR = 'outputs/'
cfg.MODEL.WEIGHTS = "detr/converted_model.pth"
cfg.MODEL.DETR.NUM_CLASSES = 1
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0000125
cfg.SOLVER.MAX_ITER = 46240
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.SOLVER.CHECKPOINT_PERIOD = number_of_iterations

# Train the model by using the trainer provided by DETR
from d2.train_net import Trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Save config to file
config_file = 'config_final_46k_125e-05_512.yaml'
with open("/home/habtamu/DETR_Multiview_without_Diff/outputs/"+config_file, "w") as f:
    f.write(cfg.dump())

# Rename the model file
os.rename("/home/habtamu/DETR_Multiview_without_Diff/outputs/model_final.pth", "/home/habtamu/DETR_Multiview_without_Diff/outputs/model_final_46k_125e-05_512.pth")
