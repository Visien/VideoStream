import argparse
import os

import torch
import numpy as np
from torchvision import transforms
import cv2

import _init_paths  # noqa: F401

from pet.lib.utils.checkpointer import get_weights, load_weights
from pet.lib.utils.comm import all_gather, init_seed, is_main_process, synchronize
from pet.lib.utils.logger import build_test_hooks
from pet.lib.utils.misc import logging_rank, mkdir_p, setup_logging
from pet.lib.utils.timer import Timer
from PIL import Image

from pet.cnn.core.inference import Inference
from pet.cnn.datasets.dataset import build_dataset, make_test_data_loader
from pet.cnn.datasets.postprocess import CNNPostProcessor
from pet.cnn.modeling.model_builder import GeneralizedCNN
from pet.cnn.utils.analyser import RCNNAnalyser

from pet.projects.centerrit.core.config import get_base_cfg, infer_cfg
from pet.projects.centerrit.data.structures.ritbox import ritbox2poly
import time


def vis_ritbox(im, ritbox, bbox_color):
    """Visualizes a ritbox."""
    h, w = im.shape[:2]
    poly = ritbox2poly(ritbox, (w / 2, h / 2))
    quad = [[poly[0], poly[1]], [poly[2], poly[3]], [poly[4], poly[5]], [poly[6], poly[7]]]
    quad = np.asarray([quad], dtype=np.int32)
    img = cv2.polylines(im, quad, 1, bbox_color, thickness=2)

    return img


class Fisheeye_Inference(object):
    def __init__(self):

        self.cfg = get_base_cfg()
        self.cfg.merge_from_file('/home/user/Program/xinxueshi_workspace/Pet-dev/ckpts/projects/centerrit/loaf/1k/centerrit_MV3-LM-0.5-XD128@D2-WODCN-ST1-SYNCBN128_adam_0.25x/centerrit_MV3-LM-0.5-XD128@D2-WODCN-ST1-SYNCBN128_adam_0.25x.yaml')
        # self.cfg.merge_from_list(args.opts)
        self.cfg = infer_cfg(self.cfg)
        self.cfg.freeze()

        init_seed(self.cfg.MISC.SEED)

        torch.cuda.set_device(0)

    # Create model
        self.model = GeneralizedCNN(self.cfg)

    # Load model
        self.test_weights = get_weights(self.cfg.MISC.CKPT, self.cfg.TEST.WEIGHTS)
        load_weights(self.model, self.test_weights)
        self.model.eval()
        self.model.to(torch.device(self.cfg.MISC.DEVICE))

    # Build test engine
        self.inference = Inference(self.cfg, self.model)

    def __call__(self, img):
        time1 = time.time()
        # img = Image.open('/home/user/Program/xinxueshi_workspace/Pet-dev/tools/projects/centerrit/resource/00.jpg')
        result = self.inference(img)
        time3 = time.time()

        boxes = result[0][0]
        print(len(boxes))
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        vis_img = None

        for i in sorted_inds:
            bbox = boxes[i, :-1]
            score = boxes[i, -1]
            if score < 0.5:
                continue
            if len(boxes) == 0:
                vis_img = img
            else:
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                vis_img = vis_ritbox(img, bbox, (0,0,0))
        time2 = time.time()
        print(time3-time1)
        print(time2-time3)

        return vis_img
        # cv2.imwrite('/home/user/Program/xinxueshi_workspace/Pet-dev/tools/projects/centerrit/resource/00_test.jpg',vis_img)


if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser(description="Pet Model Testing")
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--cfg",
    #                     type=str,
    #                     dest="cfg_file",
    #                     default="cfgs/projects/centerrit/centerrit_DLA-34_adam_aug.yaml",
    #                     help="optional config file")
    # parser.add_argument("opts",
    #                     nargs=argparse.REMAINDER,
    #                     help="See pet/projects/centerrit/core/config.py for all options")
    #
    # args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    # main(args)
    model = Fisheeye_Inference()
    img = Image.open('/home/user/Program/xinxueshi_workspace/Pet-dev/tools/projects/centerrit/resource/00.jpg')
    vis_img = model(img)
    cv2.imwrite('/home/user/Program/xinxueshi_workspace/Pet-dev/tools/projects/centerrit/result/00_res.jpg', vis_img)

