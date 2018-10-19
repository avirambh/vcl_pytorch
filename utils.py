import numpy as np
from my_augmentations import zero_pad, random_crop, horizontal_flip


def run_augmentation_v1(image):
    image = zero_pad(image, (40,40,3), 4)
    image = random_crop(image, 32)
    image = horizontal_flip(image, 0.5)
    return image


def load_data(path, test=False):
    data = []
    labels = []
    for slice in range(1,6):
        if test:
            data_dict = np.load(path + 'test_batch')
        else:
            data_dict = np.load(path + 'data_batch_' + str(slice))
        data.extend(data_dict['data'].
                    reshape((len(data_dict['data']), 3, 32, 32)).
                    transpose(0, 2, 3, 1))
        labels.extend(data_dict['labels'])
        if test: break
    return data, labels


def change_lr(optimizer, lr):
    """
    Looping over the parameter groups and changing their learning rate

    :param optimizer: The optimizer to loop over
    :param lr: the new learning rate
    """
    for g in optimizer.param_groups:
        g['lr'] = lr
        print "LR CHANGED: ", g['lr']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count