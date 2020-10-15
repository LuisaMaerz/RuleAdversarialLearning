import torch.nn as nn


class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, number_of_classes, linear=False):
        super(SingleLayerClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_extractor = FeatureExtractor(input_size, hidden_size, linear=linear)
        self.classifier = Classifier(self.hidden_size, number_of_classes, linear=linear)

        def forward(self, x):
            features = self.feature_extactor(x)
            output = self.classifier(features)
            return output


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, linear=False):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_layer = nn.Linear(self.input_size, self.hidden_size)
        self.is_linear = linear
        if not linear:
            self.relu = nn.ReLU()

    def forward(self, x):
        projection = self.linear_layer
        if self.is_linear:
            return projection

        features = self.relu(projection)
        return features


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes, linear=False):
        super(Classifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.linear_layer = nn.Linear(feature_dim, num_classes)
        self.is_linear
        if not self.is_linear:
            self.relu = nn.Relu()

        def forward(self, x):
            projection = self.linear_layer
            if self.is_linear:
                return projection
            classification = self.relu(projection)
            return classification
