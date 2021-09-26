import paddle
import paddle.nn as nn
import paddle.nn.initializer#\

#nn.Conv1D.weight

class BaseColor(nn.Layer):
    def __init__(self):
        super(BaseColor, self).__init__()

        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

