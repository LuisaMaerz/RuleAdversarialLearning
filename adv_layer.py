import torch.nn as nn
from torch.autograd import Function


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


class Discriminator(nn.Module):
    def __init__(self, size_factor, num_ents):
        super(Discriminator, self).__init__()
        self.size_factor = size_factor
        self.num_classes = num_ents
        #self.ent_classifier = nn.Sequential()
        self.ent_classifier = nn.Linear(size_factor, num_ents) #size_factor * 2
        #self.ent_classifier.add_module('predict_ents', nn.Linear(size_factor * 2, num_ents))


    def forward(self, x):
        #x = grad_reverse(x)
        x = self.ent_classifier(x)
        return x

