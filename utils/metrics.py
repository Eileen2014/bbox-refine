import numpy as np
import pickle
import os

from utils.options import options
from utils.common import file_exists
from utils.common import join

from chainercv.evaluations import eval_detection_voc

class Evaulate:
    def __init__(self, opts):
        self.opts = opts

    def pprint(self, det_stats, ref_stats, iou_thresh):
        lb_names = self.opts['label_names']

        formatter = '{:^5}{:^20}{:^20}{:^20}'
        print('IoU thresh: {:.2f}'.format(iou_thresh))
        print(formatter.format('Idx', 'Class', 'Detector AP', 'Refined AP'))
        print('-'*65)
        for idx, (class_name, det_AP, ref_AP) in enumerate(zip(lb_names, det_stats['ap'], ref_stats['ap'])):
            print(formatter.format(idx, class_name, det_AP, ref_AP))
        print('Overall Stats: Detector mAP: {:.2f}, Refined mAP: {:.2f}'.format(det_stats['map'], ref_stats['map']))
        print('-'*65)

    def evaulate_single_model(self, logsdir):
        assert file_exists(join([logsdir, 'metrics.list'])), "Metrics file doesn't exist!"

        with open(join([logsdir, 'metrics.list']), 'rb') as f:
            metrics = pickle.load(f)
        
        pred_bboxes, pred_labels, pred_scores, refn_bboxes, gt_bboxes, gt_labels = zip(*metrics.values())

        for iou_thresh in np.linspace(0.5, 1.0, 11):
            detector_stats = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                iou_thresh=iou_thresh
                )
            refined_stats = eval_detection_voc(
                refn_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                iou_thresh=iou_thresh
                )

            self.pprint(detector_stats, refined_stats, iou_thresh)

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    evaluater = Evaulate(opts)

    logsdir = opts['logs_root']
    evaluater.evaulate_single_model(logsdir)
