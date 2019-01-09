from models.detector import Detector
from utils.visualizer import Visualize
from utils.options import options
from utils.voc_loader import voc_loader
from utils.common import mkdirs
from utils.common import join

from chainercv.utils import read_image
from chainercv.utils import mask_iou
from chainercv.utils import bbox_iou
from chainercv.utils import mask_to_bbox
from chainercv.utils.bbox.non_maximum_suppression import non_maximum_suppression as nms

import os
import pickle
import sys
import time
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

        # please change me later
        self.year             = '2007'

        self.detector = Detector(
            self.detector_name,
            self.n_classes,
            self.pretrained_model,
            self.opts['gpu_id']
            )
        self.visualizer = Visualize(opts['label_names'])
        self.loader = voc_loader(data_dir=self.data_dir, split=self.split, super_root=self.super_root, year=self.year)

    def is_valid_box(self, bbox_mask):
        """ Checks if the box is valid
            Just a sanity check, some bboxes were invalid
        """
        return np.any(bbox_mask)

    def box_to_mask(self, box, size):
        """Convert box co-ordinates into a mask"""
        H, W = size
        mask_img = np.zeros((H, W), dtype=np.bool)
        y_min, x_min, y_max, x_max = box.astype(np.int32)
        # FIXME: Should be adding `y_max+1` here?
        mask_img[y_min:y_max, x_min:x_max] = True
        return mask_img

    def SD_metric(self, img, bbox, masks, stype=0):
        _s_in, _s_st = [], []
        for mask in masks:
            intersect = np.bitwise_and(bbox, mask).sum()
            ratio = intersect / np.count_nonzero(mask)
            if ratio == 1:
                _s_in.append(mask)
            elif ratio < 1:
                _s_st.append(mask.astype(np.bool))
        return np.array(_s_in), np.array(_s_st)

    def rebase_sst(self, s_in, s_st, bboxes):
        _sst = []
        for sin, sst, bbox in zip(s_in, s_st, bboxes):
            n, h, w = sst.shape
            union_masks = np.empty((n, h, w), dtype=np.float32)
            for idx, s_mask in enumerate(sst):
                union_masks[idx] = np.bitwise_or(sin, s_mask)
            union_bboxes = mask_to_bbox(union_masks)
            iou = np.squeeze(bbox_iou(union_bboxes, np.array([bbox])))
            order = np.argsort(iou, axis=0)[::-1]
            _sst.append(sst[order])
        return _sst

    def get_initial_sets(self, img, bboxes, masks, boxes):
        c, h, w = img.shape 
        s_in, s_st = [], []
        for box in bboxes:
            box_mask = self.box_to_mask(box, (h, w))
            if not self.is_valid_box(box_mask):
                continue
            _s_in, _s_st = self.SD_metric(img, box_mask, masks, stype=-1)
            if len(_s_in) == 0:
                continue
            _s_in = np.sum(np.array(_s_in), axis=0).astype(np.bool)
            s_in.append(_s_in)
            s_st.append(_s_st) 
        return s_in, s_st

    def predict_single(self, img):
        bboxes, labels, scores = self.detector.predict([img])
        self.visualizer(img, bboxes[0], labels[0], scores[0])

    def box_alignment(self, img, bboxes, masks, boxes):
        s_in, s_st = self.get_initial_sets(img, bboxes, masks, boxes)
        
        if len(s_in) == 0 or len(s_st) == 0:
            return [], []

        s_st = self.rebase_sst(s_in, s_st, bboxes)
        final_boxes = []
        final_masks = []
        for bbox, sin, sst in zip(bboxes, s_in, s_st):
            s = sin
            if s.ndim == 0:
                continue
            proc = 0 
            for sk in sst:
                new_s = np.bitwise_or(s, sk)
                iou_old = bbox_iou(mask_to_bbox(np.array([s])), np.array([bbox]))[0][0]
                iou_new = bbox_iou(mask_to_bbox(np.array([new_s])), np.array([bbox]))[0][0]
                if iou_old > iou_new:
                    break
                proc += 1
                s = new_s
            final_masks.append(s)
            final_boxes.append(mask_to_bbox(np.array([s]))[-1])
        final_masks, final_boxes = np.array(final_masks), np.array(final_boxes) 
        return final_boxes, final_masks

    def multi_thresholding_superpixel_merging(self, 
            initial_boxes, aligned_boxes, aligned_masks,
            s_masks, s_boxes, threshold=None
            ):
        """ 1. performs multi-thresholding step for different thresholds
            2. incorporate some randomness by scoring these randomly
            3. remove redundant boxes using non-maximum suppression

        args:
            initial_boxes: bboxes predicted from detector
            aligned_boxes: bboxes after bbox-alignment
            aligned_masks: masks  after bbox-alignment
            s_masks      : masks corresponding to superpixels
            s_boxes      : bounding boxes for the corresponding superpixels
            threshold    : straddling expansion threshold
        """
        def get_thresholded_spixels(threshold, s_masks, a_bbox):
            """ generates the set of superpixels which have greater than `threshold`
                overlap with the `a_bbox`
            """
            req_masks = []
            for mask in s_masks:
                intersect = np.bitwise_and(a_bbox, mask).sum()
                ratio = intersect / np.count_nonzero(mask)
                if ratio >= threshold:
                    req_masks.append(mask)
            return np.array(req_masks).astype(np.bool)

        # generate sets for different thresholds
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

        # score the boxes
        for box_set in final_set_.values():
            for idx, (thresh, seg_box) in enumerate(box_set.items()):
                r = np.random.rand()
                score = r * (idx + 1)
                box_set.update({thresh: seg_box + [score]})

        # nms
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
            save=false
            )

    def predict_all(self, inputs=None):
        metrics = {}
        print('evaluating a total of {} images'.format(self.loader.len()))
        time_taken = []
        box_align_time = []
        begin_time = time.time()
        # total_size = 10
        total_size = self.loader.len()
        invalid_count = 0
        for idx in range(self.loader.len()):
        # for idx in range(10):
            # progress bar
            done_l = (idx+1.0) / total_size
            per_done = int(done_l * 30)
            args = ['='*per_done, ' '*(30-per_done-1), done_l*100]
            sys.stdout.write('\r')
            sys.stdout.write('[{}>{}]{:.0f}%'.format(*args))
            sys.stdout.flush()

            # load images and ground truth stuff
            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(idx)

            # use the detector and predict bounding boxes
            start_time = time.time() 
            p_bboxes, p_labels, p_scores = self.detector.predict([img])
            p_bboxes = np.rint(p_bboxes[0])
            time_taken.append(time.time()-start_time)

            # box-alignment
            start_time = time.time() 
            final_bboxes, final_masks = self.box_alignment(img, p_bboxes, masks, boxes)
            box_align_time.append(time.time()-start_time) 
           
            if len(final_bboxes) == 0 or len(final_masks) == 0:
                invalid_count += 1
                continue

            # store the results in a file
            metrics.update({'{}'.format(self.loader.ids[idx]): [p_bboxes, p_labels, p_scores, final_bboxes, bboxes, labels]})
            img_file = os.path.join(self.opts['project_root'], 'logs', self.logs_root, 'qualitative', str(self.loader.ids[idx]))
            self.visualizer.box_alignment(img, p_bboxes, final_bboxes, final_masks, contours, save=True, path=img_file)
       
        print('\nTotal time taken for detection per image: {:.3f}'.format(np.mean(time_taken)))
        print('Total time taken for box alignment per image: {:.3f}'.format(np.mean(box_align_time)))
        print('Total time elapsed: {:.3f}'.format(time.time()-begin_time))
        print('Total invalid images encountered {:4d}/{:4d}'.format(invalid_count, total_size))
        with open(join([self.logs_root, 'metrics.list']), 'wb') as f:
            pickle.dump(metrics, f)

    def predict_from_file(self, detections_file, annotations_file):
        metrics = {}

        with open(detections_file, 'r') as dets:
            dets = np.load(dets)

        with open(annotations_file, 'r') as anns:
            ress = np.load(anns)

        print('evaluating a total of {} images'.format(len(ress)))
        time_taken = []
        box_align_time = []
        begin_time = time.time()
        total_size = len(ress)
        invalid_count = 0
        for idx in range(len(ress)):
            # progress bar
            done_l = (idx+1.0) / total_size
            per_done = int(done_l * 30)
            args = ['='*per_done, ' '*(30-per_done-1), done_l*100]
            sys.stdout.write('\r')
            sys.stdout.write('[{}>{}]{:.0f}%'.format(*args))
            sys.stdout.flush()

            # load images and ground truth stuff
            img, bboxes, labels, contours, masks, boxes = self.loader.load_single(idx)

            # use the detector and predict bounding boxes
            start_time = time.time() 
            p_bboxes, p_labels, p_scores = self.detector.predict([img])
            p_bboxes = np.rint(p_bboxes[0])
            time_taken.append(time.time()-start_time)

            # box-alignment
            start_time = time.time() 
            final_bboxes, final_masks = self.box_alignment(img, p_bboxes, masks, boxes)
            box_align_time.append(time.time()-start_time) 
           
            if len(final_bboxes) == 0 or len(final_masks) == 0:
                invalid_count += 1
                continue

            # store the results in a file
            metrics.update({'{}'.format(self.loader.ids[idx]): [p_bboxes, p_labels, p_scores, final_bboxes, bboxes, labels]})
            img_file = os.path.join(self.opts['project_root'], 'logs', self.logs_root, 'qualitative', str(self.loader.ids[idx]))
            self.visualizer.box_alignment(img, p_bboxes, final_bboxes, final_masks, contours, save=True, path=img_file)
       
        print('\nTotal time taken for detection per image: {:.3f}'.format(np.mean(time_taken)))
        print('Total time taken for box alignment per image: {:.3f}'.format(np.mean(box_align_time)))
        print('Total time elapsed: {:.3f}'.format(time.time()-begin_time))
        print('Total invalid images encountered {:4d}/{:4d}'.format(invalid_count, total_size))
        with open(join([self.logs_root, 'metrics.list']), 'wb') as f:
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
    elif opts['evaluate_from_file']:
        baseline.predict_from_file(opts['detections_file'], opts['annotations_file'])
    elif opts['benchmark']:
        pass
    else:
        print('Option not recognized!')
