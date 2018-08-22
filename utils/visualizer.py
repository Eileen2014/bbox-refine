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

    def box_alignment(self, img, old_boxes, new_boxes, Spixels,
                      contours, save=True, path=None
                      ):
        """ Visualize the change in the bboxes after box alignment

        Args:
            img      : Input image
            old_boxes: Old bboxes
            new_boxes: Updated bboxes
            Spixels  : set of superpixels
        """
        # fig = plt.figure(figsize=(16, 7))
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        plt.xticks([])
        plt.yticks([])
        plt.title('Superpixels')
        plt.imshow(contours)

        ax1 = fig.add_subplot(132)
        plt.xticks([])
        plt.yticks([])
        plt.title('Original box')
        vis_bbox(img, old_boxes, ax=ax1, linewidth=1.0)

        ax2 = fig.add_subplot(133)
        plt.xticks([])
        plt.yticks([])
        plt.title('After box-alignment')
        vis_bbox(img, new_boxes, ax=ax2, linewidth=1.0)
        for spixel in Spixels:
            plt.imshow(spixel, cmap='gray', alpha=0.5, interpolation='none')
        if save:
            if path is None:
                raise ValueError('Path should be set')
            plt.savefig(path+'.png')
        else:
            plt.show()
        plt.close(fig)

    def mtsm(self, img, old_boxes, new_boxes, Spixels,
            contours, s_boxes, s_superpixels,
            save=True, path=None
            ):
        """ Visualize the change in the bboxes after box alignment
            and straddling expansion

        Args:
            img      : Input image
            old_boxes: Old bboxes
            new_boxes: Updated bboxes
            Spixels  : set of superpixels
        """
        # fig = plt.figure(figsize=(16, 7))
        fig = plt.figure()
        ax1 = fig.add_subplot(141)
        plt.xticks([])
        plt.yticks([])
        plt.title('Superpixels')
        plt.imshow(contours)

        ax2 = fig.add_subplot(142)
        plt.xticks([])
        plt.yticks([])
        plt.title('Original box')
        vis_bbox(img, old_boxes, ax=ax2, linewidth=1.0)

        ax3 = fig.add_subplot(143)
        plt.xticks([])
        plt.yticks([])
        plt.title('After box-alignment')
        vis_bbox(img, new_boxes, ax=ax3, linewidth=1.0)
        for spixel in Spixels:
            plt.imshow(spixel, cmap='gray', alpha=0.5, interpolation='none')

        ax4 = fig.add_subplot(144)
        plt.xticks([])
        plt.yticks([])
        plt.title('After straddling expansion')
        vis_bbox(img, s_boxes, ax=ax4, linewidth=1.0)
        for spixel in s_superpixels:
            plt.imshow(spixel, cmap='gray', alpha=0.5, interpolation='none')

        if save:
            if path is None:
                raise ValueError('Path should be set')
            plt.savefig(path+'.png')
        else:
            plt.show()
        plt.close(fig)
