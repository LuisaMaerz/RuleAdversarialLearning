import torch.nn as nn

import models.feed_forward_blocks as ffb


class JointModel(nn.Module):
    def __init__(self, input_size, hidden_size, feature_dim, num_classes, num_patterns):
        super(JointModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature = ffb.FeatureExtractor(input_size, hidden_size)
        self.classifier = ffb.Classifier(feature_dim, num_classes)
        self.ent_classifier = ffb.Classifier(feature_dim, num_patterns)

    def forward(self, x, alpha=1):
        x = self.feature(x)
        x = x.view(-1, self.hidden_size)

        rel_pred = self.classifier(x)
        ent_pred = self.ent_classifier(x)

        return rel_pred, ent_pred


# FB: For debugging, remove afterwards...
class AdversarialModel(nn.Module):
    def __init__(self, input_size, hidden_size, feature_dim, num_classes, num_patterns):
        super(AdversarialModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature = ffb.FeatureExtractor(input_size, hidden_size)
        self.classifier = ffb.Classifier(feature_dim, num_classes)
        self.ent_classifier = ffb.Classifier(feature_dim, num_patterns)
        #self.flipper = GradientFlipper()

    def forward(self, x, alpha=1):
        x = self.feature(x)
        x = x.view(-1, self.hidden_size)
        rel_pred = self.classifier(x)
        #x_flipped = self.flipper(x, alpha)
        #ent_pred = self.ent_classifier(x_flipped)
        ent_pred = self.ent_classifier(x)
        return rel_pred, ent_pred
