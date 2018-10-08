import argparse
import os

from utils.common import join

from datasets.voc import names as voc_names

class options(object):
    """ Holds the different hyper-parameters """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.user = os.environ.get('USER')

    def initialize(self):
        # Training
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--project_root', type=str, default='', help='Path to root of the project')
        self.parser.add_argument('--data_dir', type=str, default='datasets/VOC2012',help='Path to the dataset')
        self.parser.add_argument('--split', type=str, default='train',help='Type of split for the dataset')
        self.parser.add_argument('--train', action='store_true', help='Train / test')
        self.parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs to train the model')
        self.parser.add_argument('--img_h', type=int, default=224, help='Image hieght')
        self.parser.add_argument('--img_w', type=int, default=224, help='Image width')
        self.parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
        self.parser.add_argument('--dataset', type=str, default='voc', help='Dataset to use')
        self.parser.add_argument('--detector', type=str, default='faster_rcnn', help='bbox detector')
        self.parser.add_argument('--threshold', type=float, default=-1, help='Straddling expansion threshold. -1 represents NMS is applied for thresholds in {0.1, 0.2, ..., 0.5}')
        self.parser.add_argument('--super_type', type=str, default='', help='Type of superpixels to be used')
        self.parser.add_argument('--pretrained_model', type=str, default='voc0712', help='pretrained model')
        self.parser.add_argument('--n_classes', type=int, default=21, help='Number of fore-ground classes')
        self.parser.add_argument('--ckpt_frq', type=int, default=10, help='Checkpoint frequency (in epochs)')
        self.parser.add_argument('--sample_frq', type=int, default=1, help='Sample images frequency (in epochs)')
        self.parser.add_argument('--display_frq', type=int, default=200, help='Display log after')
        self.parser.add_argument('--base_lr', type=float, default=3e-4, help='Initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Drop lr by this factor after `lr_decay_frq`')
        self.parser.add_argument('--lr_start_epoch', type=int, default=400, help='No. of epochs to train with starting learning rate')
        self.parser.add_argument('--lr_decay_frq', type=int, default=200, help='Decay the learning rate after these many epochs linearly')

        # Mode
        self.parser.add_argument('--demo', action='store_true', help='Run a demo?')
        self.parser.add_argument('--evaluate', action='store_true', help='Evaluate a model?')
        self.parser.add_argument('--benchmark', action='store_true', help='Benchmark a model?')

    def update_opts(self):
        # Datasets
        self.opts.update({'datasets': ('voc', )})

        # Detectors
        self.opts.update({'detectors': ('ssd', 'yolov2', 'yolov3', 'ssd300', 'ssd512', 'faster_rcnn')})

        # Update parameters specific to dataset
        names_map = {
            'voc': voc_names
        }

        assert self.opts['dataset'] in self.opts['datasets'], "{} dataset doesn't exist".format(self.opts['dataset'])
        assert self.opts['detector'] in self.opts['detectors'], "{} detector doesn't exist".format(self.opts['detector'])

        self.opts.update({'label_names': names_map[self.opts['dataset']]})
        self.opts.update({'n_classes': len(self.opts['label_names'])})

        # Some directories
        logs_root = join([self.opts['project_root'], 'logs', 'Detector-{}_SuperType-{}'.format(self.opts['detector'], self.opts['super_type'])])
        self.opts.update({'logs_root': logs_root})

    def parse(self, train_mode=False):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        args.train = train_mode

        self.opts = vars(args)
        self.update_opts()

        return self.opts
