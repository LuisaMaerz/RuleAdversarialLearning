import torch.nn as nn
import adv_layer
from torch.autograd import Function


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.hidden = nn.Linear(6,5)
        #self.relu = nn.ReLU()

    def forward(self, x):
        return self.hidden(x) #self.hidden(self.relu(x))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # FB: Why is number of classes in here?
        self.num_classes = 2
        self.class_classifier = nn.Linear(5, 2)

    def forward(self, x):
        return self.class_classifier(x)


class EntClassifier(nn.Module):
    def __init__(self):
        super(EntClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc1(x)
        return x


class GradientFlipper(nn.Module):
    def __init__(self):
        super(GradientFlipper, self).__init__()

    def forward(self, x, alpha = 1):
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


class DANN3(nn.Module):
    def __init__(self):
        super(DANN3, self).__init__()
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.ent_classifier = EntClassifier()
        self.flipper = GradientFlipper()

    def forward(self, x, alpha=1):
        x = self.feature(x)
        x = x.view(-1, 5)

        rel_pred = self.classifier(x)

        x_flipped = self.flipper(x, alpha)

        ent_pred = self.ent_classifier(x_flipped)

        return rel_pred, ent_pred


class DANN(nn.Module):

    def __init__(self):
        super(DANN, self).__init__()
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.ent_classifier = adv_layer.Discriminator(5, 5)  #3, da intern in adv_layer der erste Parameter verdoppelt wird


    def forward(self, x, ent_types, alpha=1):
        feature = self.feature(x)
        feature = feature.view(-1, 5)  #-1, 6
        rel_pred = self.classifier(feature)
        ent_pred = self.get_adversarial_result(feature, ent_types, alpha)

        return rel_pred, ent_pred


    def get_adversarial_result(self, x, ents, alpha=1):
        loss_fn = nn.CrossEntropyLoss()

        x = adv_layer.ReverseLayerF.apply(x, alpha)
        ent_pred = self.ent_classifier(x)

        loss_adv = loss_fn(ent_pred, ents)

        return loss_adv



class DANN2(nn.Module):

    def __init__(self):
        super(DANN2, self).__init__()
        self.num_classes_ent = 2
        self.class_classifier = nn.Linear(6, 2)

        self.ent_classifier = nn.Linear(3 * 2, 5)


    def forward(self, input, ents, alpha=1):

        feature = input.view(-1, 6)
        rel_pred = self.class_classifier(feature)
        ent_pred = self.get_adversarial_result(feature, ents, alpha)

        return rel_pred, ent_pred


    def get_adversarial_result(self, x, ents, alpha=1):
        loss_fn = nn.CrossEntropyLoss()

        x = adv_layer.ReverseLayerF.apply(x, alpha)
        ent_pred = self.ent_classifier(x)

        loss_adv = loss_fn(ent_pred, ents)

        return loss_adv
