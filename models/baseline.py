from models.detector import Detector
from utils.visualizer import Visualize
from utils.options import options

from chainercv.utils import read_image

class Baseline:
    def __init__(self, opts):
        self.opts = opts

        self.n_classes        = opts['n_classes']
        self.pretrained_model = opts['pretrained_model']
        self.detector_name    = opts['detector']

        self.detector = Detector(
            self.detector_name,
            self.n_classes,
            self.pretrained_model,
            self.opts['gpu_id']
            )
        self.visualizer = Visualize(opts['label_names'])
    
    def predict_single(self, img):
        bboxes, labels, scores = self.detector.predict([img])
        self.visualizer(img, bboxes[0], labels[0], scores[0])

    def predict(self, imgs):
        pass

if __name__ == '__main__':
    opts = options().parse(train_mode=False)
    baseline = Baseline(opts)
    img = read_image('../utils/imgs/sample.jpg')
    baseline.predict_single(img)
