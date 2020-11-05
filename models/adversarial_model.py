import torch.nn as nn
from torch.autograd import Function

import models.feed_forward_blocks as ffb


class GradientFlipper(nn.Module):
    def __init__(self):
        super(GradientFlipper, self).__init__()

    def forward(self, x, alpha=1):
        x = ReverseLayerF.apply(x, alpha)
        return(x)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        #print('################forward')
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print('backward###############')
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x):
    return ReverseLayerF.apply(x)


class AdversarialModel(nn.Module):
    def __init__(self, input_size, hidden_size, feature_dim, num_classes, num_patterns):
        super(AdversarialModel, self).__init__()
        self.hidden_size = hidden_size

        self.feature = ffb.FeatureExtractor(input_size, hidden_size)
        self.classifier = ffb.Classifier(feature_dim, num_classes)
        self.ent_classifier = ffb.EntClassifier(feature_dim, num_patterns)
        self.flipper = GradientFlipper()

    def forward(self, x, alpha=1):
        x = self.feature(x)
        x = x.view(-1, self.hidden_size)

        rel_pred = self.classifier(x)

        x_flipped = self.flipper(x, alpha)

        ent_pred = self.ent_classifier(x_flipped)

        return rel_pred, ent_pred
