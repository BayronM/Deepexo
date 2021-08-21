#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils import visualizer





def show_random_example(data,dataset_dicts,metadata,n_examples):
    for d in random.sample(dataset_dicts,n_examples):
        a = data[d["image_id"]]
        visualizer = Visualizer((a*255).astype(np.uint8),metadata)
        out = visualizer.draw_dataset_dict(d)
        #for planet in d["annotations"]:
        #    out = visualizer.draw_polygon(planet["segmentation"],color='g')
        plt.imshow(out.get_image(),cmap='RdBu_r')
        plt.show()
