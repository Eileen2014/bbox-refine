from models.detector import Detector
from utils.visualizer import Visualize
from utils.options import options
from utils.voc_loader import voc_loader
from utils.common import mkdirs

from chainercv.utils import read_image
from chainercv.utils import mask_iou
from chainercv.utils import bbox_iou
from chainercv.utils import mask_to_bbox
from chainercv.utils.bbox.non_maximum_suppression import non_maximum_suppression as nms

import os
import pickle
import sys
import numpy as np

class Baseline:
    def __init__(self, opts):
        self.opts = opts

        self.n_classes        = opts['n_classes']
        self.pretrained_model = opts['pretrained_model']
        self.detector_name    = opts['detector']
        self.threshold        = opts['threshold']
        self.data_dir         = os.path.join(opts['project_root'], '..', opts['data_dir'])
        self.split            = opts['split']
        self.super_type       = opts['super_type']
        self.logs_root        = opts['logs_root']
        self.project_root     = opts['project_root']
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

    def SD_metric(self, bbox, masks, stype=0):
        _s_in, _s_st = [], []
        for mask in masks:
            intersect = np.bitwise_and(bbox, mask).sum()
            ratio = intersect / np.count_nonzero(mask)
            if ratio == 1:
                _s_in.append(mask)
            elif ratio < 1:
                _s_st.append(mask.astype(np.bool))
        _s_in = np.sum(np.array(_s_in), axis=0).astype(np.bool)
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
            if S.ndim == 0:
                continue
            for sk in Sst:
                new_S = np.bitwise_or(S, sk)
                IoU_old = bbox_iou(mask_to_bbox(np.array([S])), np.array([bbox]))
                IoU_new = bbox_iou(mask_to_bbox(np.array([new_S])), np.array([bbox]))
                if IoU_old > IoU_new:
                    break
                S = new_S
            final_masks.append(S)
            final_boxes.append(mask_to_bbox(np.array([S]))[-1])
        final_masks, final_boxes = np.array(final_masks), np.array(final_boxes) 
        return final_boxes, final_masks

    def multi_thresholding_superpixel_merging(self, 
            initial_boxes, aligned_boxes, aligned_masks,
            s_masks, s_boxes, threshold=None
            ):
        """ 1. Performs multi-thresholding step for different thresholds
            2. Incorporate some randomness by scoring these randomly
            3. Remove redundant boxes using Non-maximum Suppression

        Args:
            initial_boxes: bboxes predicted from detector
            aligned_boxes: bboxes after bbox-alignment
            aligned_masks: masks  after bbox-alignment
            s_masks      : masks corresponding to superpixels
            s_boxes      : bounding boxes for the corresponding superpixels
            threshold    : Straddling expansion threshold
        """
        def get_thresholded_spixels(threshold, s_masks, a_bbox):
            """ Generates the set of superpixels which have greater than `threshold`
                overlap with the `a_bbox`
            """
            req_masks = []
            for mask in s_masks:
                intersect = np.bitwise_and(a_bbox, mask).sum()
                ratio = intersect / np.count_nonzero(mask)
                if ratio >= threshold:
                    req_masks.append(mask)
            return np.array(req_masks).astype(np.bool)

        # Generate sets for different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        final_set_ = {}
        for idx, (a_mask, a_bbox) in enumerate(zip(aligned_masks, aligned_boxes)):
            box_set = {}
            for threshold in thresholds:
                req_superpixels = get_thresholded_spixels(threshold, s_masks, a_bbox)
                super_segment = np.sum(req_superpixels, axis=0)
                final_segment = np.bitwise_or(super_segment, a_mask)
                final_bbox    = mask_to_bbox(np.array([final_segment]))
                box_set.update({threshold: [final_segment, final_bbox]})
            final_set_.update({idx: box_set})

        # Score the boxes
        for box_set in final_set_.values():
            for idx, (thresh, seg_box) in enumerate(box_set.items()):
                R = np.random.rand()
                score = R * (idx + 1)
                box_set.update({thresh: seg_box + [score]})

        # NMS
        for key, box_set in final_set_.items():
            segments, bboxes, scores = zip(*box_set.values())
            idxs = nms(bboxes, thresh=0.9, score=scores)
            final_picks = [
                segments[idxs],
                bboxes[idxs],
                scores[idxs]
            ]
            final_set_.update({key: final_picks})
        return final_set_

    def predict(self, inputs=None):
        if inputs is None:
            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(0)
        else:
            img, bboxes, labels, contours, masks, boxes = inputs

        p_bboxes, p_labels, p_scores = self.detector.predict([img])
        p_bboxes = p_bboxes[0]

        final_bboxes, final_masks = self.box_alignment(img, bboxes, masks, boxes)
        final_set = self.multi_thresholding_superpixel_merging(p_bboxes, final_bboxes,
            final_masks, masks, boxes)
        mtsm_masks, mtsm_bboxes, _ = zip(*final_set.values())
        self.visualizer.mtsm(
            img, bboxes,
            final_bboxes, final_masks, contours,
            mtsm_bboxes,mtsm_masks, 
            save=False
            )

    def predict_all(self, inputs=None):
        metrics = {}
        for idx in range(self.loader.len()):

            # Progress bar
            done_l = (idx+1.0) / self.loader.len()
            per_done = int(done_l * 30)
            args = ['='*per_done, ' '*(30-per_done-1), done_l*100]
            sys.stdout.write('\r')
            sys.stdout.write('[{}>{}]{:.0f}%'.format(*args))
            sys.stdout.flush()

            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(idx)
            final_bboxes, final_masks = self.box_alignment(img, bboxes, masks, boxes)

            p_bboxes, p_labels, p_scores = self.detector.predict([img])
            p_bboxes = p_bboxes[0] # TODO: Not yet optimized for batch processing
            # Store the results in a file
            metrics.update({'{}'.format(self.loader.ids[idx]): [p_bboxes, p_labels, p_scores, final_bboxes, bboxes, labels]})
            img_file = os.path.join(self.opts['project_root'], 'logs', self.logs_root, 'qualitative', str(self.loader.ids[idx]))
            self.visualizer.box_alignment(img, p_bboxes, final_bboxes, final_masks, contours, save=True, path=img_file)
        with open('{}/logs/{}/metrics.list'.format(self.project_root, self.logs_root), 'wb') as f:
            pickle.dump(metrics, f)

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    req_dirs = [
            os.path.join(opts['project_root'], 'logs', opts['logs_root'],  'qualitative')
                ]
    mkdirs(req_dirs)
    baseline = Baseline(opts)

    if opts['demo']:
        img = read_image('../utils/imgs/sample.jpg')
        baseline.predict_single(img)
    elif opts['evaluate']:
        baseline.predict_all()
    elif opts['benchmark']:
        pass
    else:
        print('Option not recognized!')
