import numpy as np
import pandas as pd
import warnings
import os
import xml.etree.ElementTree as ET

from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset
from chainercv.datasets.voc.voc_utils import voc_bbox_label_names

from utils.common import get_tightest_bboxes

from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class yto_loader:
    def __init__(self, data_dir, split='val', super_root=None):
        """ Parsing of XML file is taken from ChainerCV repository
            URI: https://github.com/chainer/chainercv/blob/master/chainercv/datasets/voc/voc_bbox_dataset.py
        """
        id_list_file_train = os.path.join(
            data_dir, 'ImageSets/Main/trainYTO.txt')
        id_list_file_test = os.path.join(
            data_dir, 'ImageSets/Main/testYTO.txt')

        self.ids_train = [id_.strip() for id_ in open(id_list_file_train)]
        self.ids_test  = [id_.strip() for id_ in open(id_list_file_test)]
        self.ids = self.ids_train + self.ids_test

        self.data_dir = data_dir
        self.super_root = super_root
        self.use_difficult = True

    def len(self):
        return len(self.ids)

    def _get_image(self, i):
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _get_annotations(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        return bbox, label

    def fix_superpixels(self, contours):
        """ This is just a temporary fix
            Currently the superpixels have one less row
        """
        H, W = contours.shape
        new_contours = np.empty((H+1, W), dtype=contours.dtype)
        new_contours[:-1, :] = contours
        new_contours[-1, :] = contours[-1]
        return new_contours

    def get_superpixels(self, idx):
        contours = pd.read_csv(os.path.join(self.super_root, self.ids[idx]+'.csv')).values
        contours = self.fix_superpixels(contours)
        min_region = contours.min()
        max_region = contours.max()
        masks = np.array([contours == idx for idx in range(min_region, max_region+1, 1)])
        boxes = get_tightest_bboxes(masks)

        return contours, masks, boxes

    def load_single(self, idx):
        img = self._get_image(idx)
        bbox, labels = self._get_annotations(idx)
        contours, masks, boxes = self.get_superpixels(idx)
        return (img, bbox, labels, contours, masks, boxes)

    def load_batch(self, start_idx, end_idx):
        pass
