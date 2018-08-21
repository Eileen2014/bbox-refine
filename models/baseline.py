from models.detector import Detector
from utils.visualizer import Visualize
from utils.options import options
from utils.voc_loader import voc_loader

from chainercv.utils import read_image
from chainercv.utils import mask_iou
from chainercv.utils import bbox_iou
from chainercv.utils import mask_to_bbox

import os
import numpy as np

class Baseline:
    def __init__(self, opts):
        self.opts = opts

        self.n_classes        = opts['n_classes']
        self.pretrained_model = opts['pretrained_model']
        self.detector_name    = opts['detector']
        self.data_dir         = os.path.join(opts['project_root'], opts['data_dir'])
        self.split            = opts['split']
        self.super_root       = os.path.join(opts['project_root'], opts['data_dir'], opts['super_type'])

        self.detector = Detector(
            self.detector_name,
            self.n_classes,
            self.pretrained_model,
            self.opts['gpu_id']
            )
        self.visualizer = Visualize(opts['label_names'])
        self.loader = voc_loader(data_dir=self.data_dir, split=self.split, super_root=self.super_root)

    def SD_metric(self, bbox, masks, stype=0):
        _s_in, _s_st = [], []
        for mask in masks:
            intersect = np.bitwise_and(bbox, mask).sum()
            ratio = intersect / np.count_nonzero(mask)
            if ratio == 1:
                _s_in.append(mask)
            elif ratio < 1:
                _s_st.append(mask)
        return np.array(_s_in), np.array(_s_st)

    def rebase_Sst(self, S_in, S_st, bboxes):
        _Sst = []
        for Sin, Sst, bbox in zip(S_in, S_st, bboxes):
            N, H, W = Sst.shape
            union_masks = np.empty((N, H, W), dtype=np.float32)
            for idx, s_mask in enumerate(Sst):
                union_masks[idx] = np.bitwise_or(Sin, s_mask)
            union_bboxes = mask_to_bbox(union_masks)
            IoU = bbox_iou(union_bboxes, bbox)
            order = np.argsort(IoU, axis=0)[::-1]
            _Sst.append(Sst[order])
        return _Sst

    def get_initial_sets(self, bboxes, masks, boxes):
        S_in, S_st = [], []
        for box in bboxes:
            _s_in, _s_st = self.SD_metric(box, masks, stype=-1)
            S_in.append(_s_in)
            S_st.append(_s_st) 
        return S_in, S_st

    def predict_single(self, img):
        bboxes, labels, scores = self.detector.predict([img])
        self.visualizer(img, bboxes[0], labels[0], scores[0])

    def box_alignment(self, bboxes, masks, boxes):
        S_in, S_st = self.get_initial_sets(bboxes, masks, boxes)
        S_st = self.rebase_Sst(S_in, S_st, bboxes)
        final_boxes = []
        final_masks = []
        for bbox, Sin, Sst in zip(bboxes, Sin, S_st):
            S = Sin
            for sk in Sst:
                new_S = np.bitwise_or(S, sk)
                IoU_old = bbox_iou(mask_to_bbox(np.array([S])), bbox)
                IoU_new = bbox_iou(mask_to_bbox(np.array([new_S])), bbox)
                if IoU_old > IoU_new:
                    break
                S = new_S
            final_masks.append(S)
            final_boxes.append(mask_to_bbox(np.array([S])))
        return final_boxes, final_masks

    def multi_thresholding_superpixel_merging(self):
        pass

    def predict(self, inputs):
        img, bboxes, labels, masks, boxes = inputs
        final_bboxes, final_masks = self.box_alignment(bboxes, masks, boxes)
        self.visualizer.box_alignment(img, bboxes, final_bboxes, final_masks)


if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    baseline = Baseline(opts)
    img = read_image('../utils/imgs/sample.jpg')
    baseline.predict_single(img)
