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
        self.data_dir         = os.path.join(opts['project_root'], '..', opts['data_dir'])
        self.split            = opts['split']
        self.super_root       = os.path.join(opts['project_root'], '..', opts['data_dir'], 'superpixels', opts['super_type'])

        self.detector = Detector(
            self.detector_name,
            self.n_classes,
            self.pretrained_model,
            self.opts['gpu_id']
            )
        self.visualizer = Visualize(opts['label_names'])
        self.loader = voc_loader(data_dir=self.data_dir, split=self.split, super_root=self.super_root)

    def box_to_mask(self, box, size):
        """Convert box co-ordinates into a mask"""
        H, W = size
        mask_img = np.zeros((H, W), dtype=np.bool)
        y_min, x_min, y_max, x_max = box.astype(np.int32)
        # FIXME: Should be adding `y_max+1` here?
        mask_img[y_min:y_max, x_min:x_max] = True
        return mask_img

    def deform_bboxes(self, bboxes, size):
        _, H, W = size
        new_boxes = np.empty_like(bboxes, dtype=bboxes.dtype)
        for idx, bbox in enumerate(bboxes):
            y_min, x_min, y_max, x_max = bbox
            if y_min < 20:
                y_min += np.random.randint(5, 20)
            else:
                y_min -= np.random.randint(5, 20)
            if x_min < 20:
                x_min += np.random.randint(5, 20)
            else:
                x_min -= np.random.randint(5, 20)
            if y_max > H - 20:
                y_max -= np.random.randint(5, 20)
            else:
                y_max += np.random.randint(5, 20)
            if x_max > W - 20:
                x_max -= np.random.randint(5, 20)
            else:
                x_max += np.random.randint(5, 20)
            new_boxes[idx] = [y_min, x_min, y_max, x_max]
        return new_boxes

    def SD_metric(self, bbox, masks, stype=0):
        _s_in, _s_st = [], []
        for mask in masks:
            intersect = np.bitwise_and(bbox, mask).sum()
            ratio = intersect / np.count_nonzero(mask)
            if ratio == 1:
                _s_in.append(mask)
            elif ratio < 1:
                _s_st.append(mask)
        _s_in = np.sum(np.array(_s_in), axis=0)
        return _s_in, np.array(_s_st)

    def rebase_Sst(self, S_in, S_st, bboxes):
        _Sst = []
        for Sin, Sst, bbox in zip(S_in, S_st, bboxes):
            N, H, W = Sst.shape
            union_masks = np.empty((N, H, W), dtype=np.float32)
            for idx, s_mask in enumerate(Sst):
                union_masks[idx] = np.bitwise_or(Sin, s_mask)
            union_bboxes = mask_to_bbox(union_masks)
            IoU = np.squeeze(bbox_iou(union_bboxes, np.array([bbox])))
            order = np.argsort(IoU, axis=0)[::-1]
            print('>> >> >> ')
            print(IoU[order][:20])
            _Sst.append(Sst[order])
        return _Sst

    def get_initial_sets(self, img, bboxes, masks, boxes):
        C, H, W = img.shape 
        S_in, S_st = [], []
        for box in bboxes:
            box_mask = self.box_to_mask(box, (H, W))
            _s_in, _s_st = self.SD_metric(box_mask, masks, stype=-1)
            S_in.append(_s_in)
            S_st.append(_s_st) 
        return S_in, S_st

    def predict_single(self, img):
        bboxes, labels, scores = self.detector.predict([img])
        self.visualizer(img, bboxes[0], labels[0], scores[0])

    def box_alignment(self, img, bboxes, masks, boxes):
        S_in, S_st = self.get_initial_sets(img, bboxes, masks, boxes)
        S_st = self.rebase_Sst(S_in, S_st, bboxes)
        final_boxes = []
        final_masks = []
        for bbox, Sin, Sst in zip(bboxes, S_in, S_st):
            S = Sin
            for sk in Sst:
                new_S = np.bitwise_or(S, sk)
                IoU_old = bbox_iou(mask_to_bbox(np.array([S])), np.array([bbox]))
                IoU_new = bbox_iou(mask_to_bbox(np.array([new_S])), np.array([bbox]))
                print('IoU old: {} IoU new: {}'.format(IoU_old, IoU_new))
                if IoU_old > IoU_new:
                    break
                S = new_S
            final_masks.append(S)
            final_boxes.append(mask_to_bbox(np.array([S]))[-1])
        final_masks, final_boxes = np.array(final_masks), np.array(final_boxes) 
        return final_boxes, final_masks

    def multi_thresholding_superpixel_merging(self):
        pass

    def predict(self, inputs=None):
        if inputs is None:
            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(0)
        else:
            img, bboxes, labels, contours, masks, boxes = inputs
 
        # bboxes = self.deform_bboxes(bboxes, img.shape)
        bboxes[1][0] += 25
        bboxes[1][1] += 40
        bboxes[1][2] -= 40
        bboxes[1][3] -= 40
 
        bboxes = [bboxes[1]]
        labels = [labels[1]]
        final_bboxes, final_masks = self.box_alignment(img, bboxes, masks, boxes)
        print('Initial boxes')
        print(bboxes)
        print('After box alignment')
        print(final_bboxes)
        self.visualizer.box_alignment(img, bboxes, final_bboxes, final_masks, contours)

    def predict_all(self, inputs=None):
        for idx in range(self.loader.len()):
            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(idx)
            
            bboxes = self.deform_bboxes(bboxes, img.shape)
            final_bboxes, final_masks = self.box_alignment(img, bboxes, masks, boxes)
            print('Initial boxes')
            print(bboxes)
            print('After box alignment')
            print(final_bboxes)
            img_file = os.path.join('/home/avisek/kv/bbox_refine/test/', self.loader.ids[idx])
            self.visualizer.box_alignment(img, bboxes, final_bboxes, final_masks, contours, save=True, path=img_file)

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    baseline = Baseline(opts)
    # img = read_image('../utils/imgs/sample.jpg')
    # baseline.predict_single(img)
    # baseline.predict_all()
    baseline.predict()
