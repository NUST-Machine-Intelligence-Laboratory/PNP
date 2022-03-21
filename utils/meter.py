from easydict import EasyDict


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


class GeneralTrainMeters(object):
    def __init__(self):
        self.meters = EasyDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_accuracy'] = AverageMeter()

    def update(self, val_l, val_acc, n_l, n_acc=None):
        if n_acc is None:
            n_acc = n_l
        self.meters.train_loss.update(val=val_l, n=n_l)
        self.meters.train_accuracy.update(val=val_acc, n=n_acc)

    def reset(self, exclude=None):
        if exclude != 'train_loss':
            self.meters.train_loss.reset()
        if exclude != 'train_accuracy':
            self.meters.train_accuracy.reset()

