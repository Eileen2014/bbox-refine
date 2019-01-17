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

    def pprint(self, 
        det_stats, box_align_stats, 
        straddling_stats, iou_thresh
        ):
        
        lb_names = self.opts['label_names']

        formatter1 = '{:^5}{:^20}{:^20}{:^20}{:^20}'
        formatter2 = '{:^5}{:^20}{:^20.3f}{:^20.3f}{:^20.3f}'
        sstr = ''
        sstr += 'IoU thresh: {:.2f}'.format(iou_thresh)+'\n'
        sstr += formatter1.format('Idx', 'Class', 'Detector AP', 'Box-align AP', 'Straddling AP')+'\n'
        sstr += '-'*85+'\n'
        for idx, (class_name, det_AP, box_AP, std_AP) in enumerate(zip(lb_names, det_stats['ap'], box_align_stats['ap'], straddling_stats['ap'])):
            sstr += formatter2.format(idx, class_name, det_AP, box_AP, std_AP)+'\n'
        sstr += 'Overall Stats: Detector mAP: {:.2f}, Box-align mAP: {:.2f}, Straddling mAP: {:.2f}'.format(det_stats['map'], box_align_stats['map'], straddling_stats['map'])+'\n'
        sstr += '-'*85+'\n'
        print(sstr)
        return sstr

    def get_list_of_metrics(self, metrics):
        """ Generate list of metrics
        """
        pred_bboxes, pred_labels, pred_scores, refn_bboxes, gt_bboxes, gt_labels, mtsm_bboxes = metrics
        p_b, p_l, p_s, r_b, g_b, g_l, m_b = [], [], [], [], [], [], []
        for pb, pl, ps, rb, gb, gl, mb in zip(pred_bboxes, pred_labels, pred_scores, refn_bboxes, gt_bboxes, gt_labels, mtsm_bboxes):
            if np.any(pb.shape != rb.shape):
                continue
            p_b.append(pb)
            r_b.append(rb)
            g_b.append(gb)
            g_l.append(gl)
            p_l.append(pl[0])
            p_s.append(pl[0])
            m_b.append(np.rint(mb))
        return [p_b, p_l, p_s, r_b, g_b, g_l, m_b]

    def evaulate_single_model(self, logsdir):
        metrics_file = join([logsdir, 'metrics.list']) 
        assert file_exists(metrics_file), "Metrics file {} doesn't exist!".format(metrics_file)

        with open(join([logsdir, 'metrics.list']), 'rb') as f:
            metrics = pickle.load(f)
 
        p_b, p_l, p_s, r_b, g_b, g_l, m_b = zip(*metrics.values())
        pred_bboxes, pred_labels, pred_scores, refn_bboxes, gt_bboxes, gt_labels, mtsm_bboxes = self.get_list_of_metrics(
                [p_b, p_l, p_s, r_b, g_b, g_l, m_b]
                )

        for iou_thresh in np.linspace(0.5, 1.0, 11):
            detector_stats = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                iou_thresh=iou_thresh
                )
            bbox_align_stats = eval_detection_voc(
                refn_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                iou_thresh=iou_thresh
                )
            straddling_stats = eval_detection_voc(
                mtsm_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                iou_thresh=iou_thresh
                )

            sstr = self.pprint(detector_stats, bbox_align_stats, straddling_stats, iou_thresh)
            with open(join([logsdir,  'metrics_{:.2f}.table'.format(iou_thresh)]), 'w') as f:
                f.write(sstr)

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    evaluater = Evaulate(opts)

    logsdir = opts['logs_root']
    evaluater.evaulate_single_model(logsdir)
