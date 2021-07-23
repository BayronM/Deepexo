#!/usr/bin/env python3


import detectron2
from detectron2.structures import BoxMode
import numpy as np
import os, json, cv2, random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data import detection_utils





def create_segmentation(px,py,size):
    range_px = list(range(px,px+size))
    range_py = list(range(py,py+size))
    box = []

    for i in range(0,size-1):
        box.extend([range_px[i],py])
        box.extend([range_px[i],py+(size-1)])
        box.extend([range_py[i],px])
        box.extend([range_py[i],px+(size-1)])
    return box



def get_exo_dict(data,position):

    dataset_dicts = []
    for idx in range(0,len(data)-1):
        record = {}
        height,width = data[idx].shape[0],data[idx].shape[1]
        record["filename"] = 'image_'+str(idx)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        for planet in position[idx]:
            #poly = create_segmentation(planet[0],planet[1])
            #poly = [p for x in poly for p in x]
            obj = {
                "bbox" : [planet[0]-4,planet[1]-3,planet[0]+4,planet[1]+3],
                "bbox_mode" : BoxMode.XYWH_ABS,
                #"segmentation": [poly],
                "category_id":0,
                "center": (planet[0],planet[1])
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts





#dataset_dicts = get_exo_dict(preprocessed_data,position)


def show_random_example(data,dataset_dicts,metadata,n_examples):
    for d in random.sample(dataset_dicts,n_examples):
        a = np.tile(data[d["image_id"]],3)
        visualizer = Visualizer((a*255).astype(np.uint8),metadata)
        for planet in d["annotations"]:
            out = visualizer.draw_box(planet["bbox"])
        plt.imshow(out.get_image(),cmap='RdBu_r')
        plt.show()
