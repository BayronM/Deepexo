#!/usr/bin/env python3


import numpy as np
import os
from dataset import get_exo_dict, show_random_example
from data_preprocess import data_preprocess
from detectron2.data import MetadataCatalog,DatasetCatalog,DatasetMapper,build_detection_train_loader,build_detection_test_loader
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.evaluation import COCOEvaluator,SemSegEvaluator
from detectron2.utils.visualizer import Visualizer


import copy
import torch
import random
import matplotlib.pyplot as plt

#Dataset Configuration
TRAIN_IMAGES_DIR = "../data/CNN/master_training.npy"
PSF_IMAGES = "../data/CNN/tinyPSF.npy"
TEST_IMAGES_DIR = "../data/CNN/master_test.npy"
C_RATIO = [0.05,0.1]
NAME_DATASET_TRAIN = "ExoDataset_train"
NAME_DATASET_TEST = "ExoDataset_test"


#Train Configuration
cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (NAME_DATASET_TRAIN,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.DEVICE='cpu'
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 100 # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
#------------------------------------------------


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = np.float32(np.moveaxis(dataset_dict['file'],-1,0))
    #plt.imshow(image)
    #plt.show()
    image = torch.from_numpy(image)
    annos = []
    for annotation in dataset_dict.pop("annotations"):
        annos.append(annotation)
    instances = utils.annotations_to_instances(annos, image)
    #print(instances)

    return{
        "image" : image,
        "width" : dataset_dict["width"],
        "height" : dataset_dict["height"],
        "instances": utils.annotations_to_instances(annos,image)
    }



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):
        return COCOEvaluator(dataset_name)
        #if output_folder is None:
        #    output_folder = os.path.join(cfg.OUTPUT_DIR).resume_or_load(
        #        cfg.MODEL.WEIGHTS, resume=
        #    )
    @classmethod
    def build_train_loader(cls,cfg):
        return build_detection_train_loader(cfg,mapper=custom_mapper)
    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        return build_detection_test_loader(cfg,dataset_name,mapper=custom_mapper)



def main():

    #Load the data
    print("-------------------\nTrain Data : ")
    data_train,_,_,position_train=data_preprocess(np.load(TRAIN_IMAGES_DIR),
                                            np.load(PSF_IMAGES),
                                            c_ratio=C_RATIO,
                                            no_blend = True)
    print("--------------------\nTest Data : ")
    data_test,_,_,position_test=data_preprocess(np.load(TEST_IMAGES_DIR),
                                           np.load(PSF_IMAGES),
                                           c_ratio=C_RATIO,
                                           no_blend = True)


    #Register the dataset for detectron2
    #Train dataset
    DatasetCatalog.register(NAME_DATASET_TRAIN, lambda a=data_train,b=position_train: get_exo_dict(a,b))
    MetadataCatalog.get(NAME_DATASET_TRAIN).set(thing_classes=["planet"])
    #Test dataset
    DatasetCatalog.register(NAME_DATASET_TEST, lambda a=data_test,b=position_test: get_exo_dict(a,b))
    MetadataCatalog.get(NAME_DATASET_TEST).set(thing_classes=["planet"])

    exo_metadata_train = MetadataCatalog.get(NAME_DATASET_TRAIN)
    exo_metadata_test = MetadataCatalog.get(NAME_DATASET_TEST)

    dataset_train_dict = get_exo_dict(data_train,position_train)
    dataset_test_dict = get_exo_dict(data_test,position_test)
    show_random_example(data_train,dataset_train_dict,exo_metadata_train,3)
    #a = custom_mapper(dataset_train_dict.pop())
    #print(a)

    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    for d in random.sample(dataset_test_dict,5):
        im = d["file"]
        outputs = predictor(im)
        print(outputs)
        v = Visualizer((im*255).astype(np.uint8),metadata=exo_metadata_train,scale=10.0)
        out=v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(out.get_image().shape)
        plt.imshow(out.get_image())
        plt.show()


if __name__ == "__main__":
    main()
