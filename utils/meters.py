class Meter():
    """abstract meter class"""

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError('Needs to be overriden')

    def update(self, val, n=1):
        raise NotImplementedError('Needs to be overriden')

    def get(self):
        raise NotImplementedError('Needs to be overriden')

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)


class ScalarMeter(Meter):
    def reset(self):
        self.avg = -1

    def update(self, val):
        self.avg = val

    def get(self):
        return self.avg


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: object = 1) -> object:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg
