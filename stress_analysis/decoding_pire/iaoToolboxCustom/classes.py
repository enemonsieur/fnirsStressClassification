import numpy as np


class Epoch:

    def __init__(self, epo, time, labels, fs):
        self.data = epo
        self.time = time
        self.labels = labels
        self.fs = fs

    def __str__(self):
        data = 'Data: \n%s' % self.data
        time = 'Time: \n%s' % self.time
        labels = 'Labels: \n%s' % self.labels
        fs = 'fs: \n%s' % self.fs
        return '\n'.join([data, time, labels, fs])

    def append_epoch(self, epo2):
        self.data = np.concatenate([self.data, epo2.data], axis=0)
        self.labels = np.concatenate([self.labels, epo2.labels], axis=0)
