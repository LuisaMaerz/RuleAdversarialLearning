import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from nlp_toolkit.utility import config_loader

from data_handling.make_toy_data import make_multiclass_toy_dataset, make_binary_toy_dataset
from models.feed_forward_blocks import SingleLayerClassifier, Classifier
from models.joint_model import JointModel
from models.adversarial_model import AdversarialModel
from data_handling.data_loaders import make_data_loaders
from trainer import train_joint, train_single, test_joint, test_single

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
JOINT_CONFIG = CURRENT_FILE_LOCATION + "/config/FeedForward_class_and_patterns.cfg"
FEEDFORWARD_PATTERN_CONFIG = CURRENT_FILE_LOCATION + "/config/FeedForward_rule_pattern_on_labels.cfg"
ADVERSARIAL_CONFIG = CURRENT_FILE_LOCATION + "/config/FeedForward_adversarial_model.cfg"
FEEDFORWARD_CLASS_CONFIG = CURRENT_FILE_LOCATION + "/config/FeedForward_class_on_labels.cfg"
JOINT_CONFIG_BINARY = CURRENT_FILE_LOCATION + "/config/FeedForward_class_and_patterns_binary.cfg"
ADVERSARIAL_CONFIG_BINARY = CURRENT_FILE_LOCATION + "/config/FeedForward_adversarial_model_binary.cfg"
FEEDFORWARD_CLASS_CONFIG_BINARY = CURRENT_FILE_LOCATION + "/config/FeedForward_class_on_labels_binary.cfg"

def laod_dataset_name(data_set_name):

    if data_set_name == "multiclass":
        data_train, data_test = make_multiclass_toy_dataset()

    elif data_set_name == "binary":
        data_train, data_test = make_binary_toy_dataset()

    else:
        raise (FileNotFoundError(f"Dataset {data_set_name} does not exist"))

    return data_train, data_test


def run_joint_model(model, available_joint_models, input_size, hidden_size, feature_dim, num_classes,
                        num_patterns, data_set_name, eps, g, lr, bs, pr_path, device=0):

    if model not in available_joint_models:
        raise(NotImplementedError(f"Model {model} not implemented"))

    if model == "JointModel":
        net = JointModel(input_size, hidden_size, feature_dim, num_classes, num_patterns)

    if model == "AdversarialModel":
        net = AdversarialModel(input_size, hidden_size, feature_dim, num_classes, num_patterns)

    with torch.cuda.device(device):
        net.cuda()

    data_train, data_test = laod_dataset_name(data_set_name)
    train_loader, test_loader = make_data_loaders(data_train, data_test, batch_size = bs)

    classifer_1_loss = nn.CrossEntropyLoss()
    classifier_2_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model = train_joint(net, optimizer, classifer_1_loss, classifier_2_loss, train_loader, eps, g, pr_path, device)

    test_joint(trained_model, test_loader, data_set_name, device)


def run_single_model(model, available_single_models, input_size, hidden_size, num_classes, data_set_name, eps, lr, bs, pr_path,
                     uc, ue, device=0):

    if model not in available_single_models:
        raise(NotImplementedError(f"Model {model} not implemented"))

    if model == "SingleLayerClassifier":
        if not input_size or not hidden_size:
            raise ValueError("Please spacify input and hidden size for single layer models")
        net = SingleLayerClassifier(input_size, hidden_size, num_classes, linear=False)

    with torch.cuda.device(device):
        net.cuda()

    data_train, data_test = laod_dataset_name(data_set_name)
    train_loader, test_loader = make_data_loaders(data_train, data_test, batch_size = bs, use_classes=uc, use_ents=ue)

    classifer_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)

    trained_model = train_single(net, optimizer, classifer_loss, train_loader, eps, pr_path, uc, ue, device)

    test_single(trained_model, test_loader, num_classes, data_set_name, device)


if __name__ == "__main__":

    config = config_loader.get_config(FEEDFORWARD_CLASS_CONFIG_BINARY, interpolation=True)
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    epochs = config.getint("TRAINING", "epochs")
    gamma = config.getfloat("TRAINING", "gamma")
    batch_size = config.getint("GENERAL", "batch_size")
    project_dir_path = config.get("GENERAL", "project_dir")
    num_classes = config.getint("GENERAL", "num_classes")
    model_name = config.get("ARCHITECTURE", "model")
    dataset = config.get("DATA", "dataset")
    available_joint_models = json.loads(config.get("GENERAL", "implemented_joint_models"))
    available_single_models = json.loads(config.get("GENERAL", "implemented_single_models"))
    input_size = None
    hidden_size = None
    use_classes = config.get("DATA", "use_classes")
    use_ents = config.get("DATA", "use_ents")
    device = config.getint("GENERAL", "device")

    if config.has_option("ARCHITECTURE", "input_size"):
        input_size = config.getint("ARCHITECTURE", "input_size")

    if config.has_option("ARCHITECTURE", "hidden_size"):
        hidden_size = config.getint("ARCHITECTURE", "hidden_size")

    if config.has_option("ARCHITECTURE", "feature_dim"):
        feature_dim = config.getint("ARCHITECTURE", "feature_dim")

    if config.has_option("ARCHITECTURE", "num_patterns"):
        num_patterns = config.getint("ARCHITECTURE", "num_patterns")

    if len(sys.argv) > 1:
        epochs, gamma = sys.argv[1:]
        epochs, gamma = int(epochs[0]), int(gamma[0])

    if model_name not in available_single_models + available_joint_models:
        raise(NotImplementedError(f"Unknown Model {model_name}"))

    if model_name in available_joint_models:
        run_joint_model(model_name,
                        available_joint_models, input_size, hidden_size, feature_dim, num_classes,
                        num_patterns, dataset, epochs, gamma, learning_rate, batch_size, project_dir_path, device=device)
    if model_name in available_single_models:
        run_single_model(model_name, available_single_models, input_size, hidden_size, num_classes,
                        dataset, epochs, learning_rate, batch_size, project_dir_path, use_classes, use_ents, device=device)