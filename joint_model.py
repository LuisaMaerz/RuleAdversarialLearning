import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.hidden = nn.Linear(8,5)
        #self.relu = nn.ReLU()

    def forward(self, x):
        return self.hidden(x) #self.hidden(self.relu(x))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.num_classes = 4
        self.class_classifier = nn.Linear(5, 4)

    def forward(self, x):
        return self.class_classifier(x)


class EntClassifier(nn.Module):
    def __init__(self):
        super(EntClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc1(x)
        return x


class DANN3(nn.Module):

    def __init__(self):
        super(DANN3, self).__init__()
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.ent_classifier = EntClassifier()

    def forward(self, x, alpha=1):
        x = self.feature(x)
        x = x.view(-1, 5)

        rel_pred = self.classifier(x)

        ent_pred = self.ent_classifier(x)

        return rel_pred, ent_pred
