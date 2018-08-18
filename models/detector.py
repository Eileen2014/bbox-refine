import chainer
import chainer.links as L
import chainer.functions as F

from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3

from datasets.voc import names as voc_names

class Detector:
    def __init__(self, detector, n_classes, pretrained_model, gpu_id=-1):
        self.detectors = {
            'faster_rcnn': _FasterRCNNVGG16,
            'ssd300'     : _SSD300,
            'ssd512'     : _SSD512,
            'yolov2'     : _YOLOv2,
            'yolov3'     : _YOLOv3
        }
        self.opts = {
            'n_classes'       : n_classes,
            'pretrained_model': pretrained_model
        }
        self.gpu_id = gpu_id
        self.model = self.detectors[detector](self.opts)
        if self.gpu_id >= 0:
            self.to_gpu()

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores

    def to_gpu(self):
        self.model.model.to_gpu()

class _FasterRCNNVGG16:
    def __init__(self, opts):
        self.model = FasterRCNNVGG16(
            n_fg_class=opts['n_classes'],
            pretrained_model=opts['pretrained_model'])

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores

class _SSD300:
    def __init__(self, opts):
        self.model = SSD300(
            n_fg_class=opts['n_classes'],
            pretrained_model=opts['pretrained_model'])

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores

class _SSD512:
    def __init__(self, opts):
        self.model = SSD512(
            n_fg_class=opts['n_classes'],
            pretrained_model=opts['pretrained_model'])

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores

class _YOLOv2:
    def __init__(self, opts):
        self.pretrained_models = ('voc0712', )
        assert opts['pretrained_model'] in self.pretrained_models, "{} pretrained model doesn't exist".format(opts.pretrained_model)

        self.model = YOLOv2(
            n_fg_class=opts['n_classes'],
            pretrained_model=opts['pretrained_model'])

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores

class _YOLOv3:
    def __init__(self, opts):
        self.pretrained_models = ('voc0712', )
        assert opts['pretrained_model'] in self.pretrained_models, "{} pretrained model doesn't exist".format(opts.pretrained_model)

        self.model = YOLOv3(
            n_fg_class=opts['n_classes'],
            pretrained_model=opts['pretrained_model'])

    def predict(self, imgs):
        bboxes, labels, scores = self.model.predict(imgs)
        return bboxes, labels, scores
