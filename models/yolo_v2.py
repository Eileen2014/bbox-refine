import itertools
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda

from chainercv.links import SSD512
from chainercv.links import YOLOv2

from chainercv.utils import read_image
from chainercv.utils import bbox_iou
from chainercv import transforms
from chainercv.visualizations import vis_bbox

from utils.voc_loader import voc_loader
from utils.options import options
from utils.common import mkdirs


class _YOLOv2(YOLOv2):
    """ This model extends the original model 
        to train on a dataset
    """
    def __init__(self, n_fg_class=None, pretrained_model=None):
        super(_YOLOv2, self).__init__(n_fg_class, pretrained_model)

    def forward(self, imgs):
        """ Forward batch of images and 
            predict bounding boxes
        """

        x = []
        params = []
        for img in imgs:
            _, H, W = img.shape
            img, param = transforms.resize_contain(
                img / 255, (self.insize, self.insize), fill=0.5,
                return_param=True)
            x.append(self.xp.array(img))
            param['size'] = (H, W)
            params.append(param)

        x = self.xp.stack(x)

        bboxes = []
        labels = []
        scores = []
        locs, objs, confs = self.__call__(x)

        locs = locs.array
        objs = objs.array
        confs = confs.array

        _bboxes = []
        _confs = []
        _objs = []
        for loc, obj, conf in zip(locs, objs, confs):
            raw_bbox = self._default_bbox.copy()
            raw_bbox[:, :2] += 1 / (1 + self.xp.exp(-loc[:, :2]))
            raw_bbox[:, 2:] *= self.xp.exp(loc[:, 2:])
            raw_bbox[:, :2] -= raw_bbox[:, 2:] / 2
            raw_bbox[:, 2:] += raw_bbox[:, :2]
            raw_bbox *= self.insize / self.extractor.grid

            obj = 1 / (1 + self.xp.exp(-obj))
            conf = self.xp.exp(conf)
            conf /= conf.sum(axis=1, keepdims=True)

            _bboxes.append(raw_bbox)
            _confs.append(conf)
            _objs.append(obj)

        return _bboxes, _confs, _objs

    def get_loss(self, 
                 g_bboxes, g_labels,
                 p_bboxes, p_confs, p_objs
                 ):
        """ Generate loss
        """
        b_loss = 0
        c_loss = 0
        p_loss = 0
        for g_bbox, g_label, p_bbox, p_conf, p_obj in zip(
            g_bboxes, g_labels, p_bboxes, p_confs, p_objs
            ):
            IoU = bbox_iou(g_bbox, p_bbox)
            pick = self.xp.argmax(IoU, axis=-1)
            p_bbox = p_bbox[pick]
            p_conf = p_conf[pick]
            p_obj = p_obj[pick]

            b_loss += F.sum((p_bbox - g_bbox) ** 2)
            c_loss += F.sum((p_conf - ))

    def train(self, opts, loader, optimizer):
        self.max_epochs = opts['max_epochs']
        self.train_size = loader.len()
        self.batch_size = opts['batch_size']

        if opts['gpu_id'] >= 0:
            super(_YOLOv2, self).to_gpu()
        optimizer.setup(super(_YOLOv2, self))

        for epoch in range(self.max_epochs):
            batch_loss = []
            for idx, (batch_begin, batch_end) in enumerate(zip(range(0, self.train_size, self.batch_size),
            range(self.batch_size, self.train_size, self.batch_size))):

                imgs, bboxes, labels = loader.load_batch(batch_begin, batch_end)
                p_bboxes, p_confs, p_objs = self.forward(imgs)

                loss = self.get_loss(bboxes, labels, p_bboxes, p_confs, p_objs)
                loss_data = cuda.to_cpu(loss)
                super(_YOLOv2, self).cleargrads()
                loss.backward()
                optimizer.update()

                batch_loss.append(loss_data)

if __name__ == '__main__':
    # opts = options().parse(train_mode=False)
    # req_dirs = [
    #         os.path.join(opts['project_root'], 'logs', 'yolov2')
    #             ]
    # mkdirs(req_dirs)

    yolo = _YOLOv2(
        n_fg_class=20,
        pretrained_model='voc0712'
        )

    # data_dir = os.path.join(opts['project_root'], '..', opts['data_dir'])
    # loader = voc_loader(
    #     data_dir=data_dir,
    #     split=opts['split'],
    #     super_root=None
    #     )
    # optimizer = optimizers.MomentumSGD()


    img = read_image('../imgs/sample.jpg')
    imgs = [img]
    yolo.forward(imgs)
    # yolo.train(opts, loader, optimizer)
