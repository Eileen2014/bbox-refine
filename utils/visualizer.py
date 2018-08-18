import matplotlib.pyplot as plt

from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names

class Visualize:
    def __init__(self, names):
        self.names = names
    
    def __call__(self, img, bbox, label, score, save=False, path=None):
        vis_bbox(
            img, bbox, label, score, label_names=self.names
        )
        plt.xticks([])
        plt.yticks([])

        if save:
            if path is None:
                raise ValueError('Path should be set')
            plt.savefig(path)
        else:
            plt.show()

    def bulk(self, samples):
        """ Visualize bulk of images
        """
        imgs, bboxs, labels, scores, paths = zip(*samples)
        for img, bbox, label, score, path in zip(
            imgs, bboxs, labels, scores, paths):
            self.__call__(
                img, bbox, label, score, save=True, path=path
                )
