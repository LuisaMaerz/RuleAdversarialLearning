import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from nlp_toolkit.utility import config_loader

from data_handling.make_toy_data import make_combined_toy_dataset, make_pattern_toy_dataset, \
    make_class_features_toy_dataset
from models.feed_forward_blocks import SingleLayerClassifier, Classifier
from models.joint_model import JointModel
from trainer import train_joint, train_single, test_joint, test_single

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
DANN3_CONFIG = CURRENT_FILE_LOCATION + "/config/dann3_combined_toy.cfg"
FEEDFORWARD_CONFIG = CURRENT_FILE_LOCATION + "/config/FeedForward_rule_pattern_on_labels.cfg"


def laod_dataset_name(data_set_name):

    if data_set_name == "combined_toy":
        data_train, data_test = make_combined_toy_dataset()

    elif data_set_name == "rule_patterns_toy":
        data_train, data_test = make_pattern_toy_dataset()

    elif data_set_name == "class_features_toy":
        data_train, data_test = make_class_features_toy_dataset()

    else:
        raise (FileNotFoundError(f"Dataset {data_set_name} does not exist"))

    return data_train, data_test


def run_joint_model(model, available_joint_models, data_set_name, eps, g, lr, bs, pr_path):

    if model not in available_joint_models:
        raise(NotImplementedError(f"Model {model} not implemented"))

    if model == "JointModel":
        net = JointModel()
    net.cuda()

    data_train, data_test = laod_dataset_name(data_set_name)
    train_loader, test_loader = get_loaders(data_train, data_test, batch_size = bs)

    classifer_1_loss = nn.CrossEntropyLoss()
    classifier_2_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model = train_joint(net, optimizer, classifer_1_loss, classifier_2_loss, train_loader, eps, g, pr_path)

    test_joint(trained_model, test_loader)


def run_single_model(model, available_single_models, input_size, hidden_size, num_classes, data_set_name, eps, lr, bs, pr_path):

    if model not in available_single_models:
        raise(NotImplementedError(f"Model {model} not implemented"))

    if model == "SingleLayerClassifier":
        if not input_size or not hidden_size:
            raise ValueError("Please spacify input and hidden size for single layer models")
        net = SingleLayerClassifier(input_size, hidden_size, num_classes, linear=False)

    net.cuda()

    data_train, data_test = laod_dataset_name(data_set_name)
    train_loader, test_loader = get_loaders(data_train, data_test, batch_size = bs)

    classifer_loss = nn.CrossEntropyLoss()
    # TODO: Put optimizer into config file
    # TODO: Put weight decay into config file
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)

    trained_model = train_single(net, optimizer, classifer_loss, train_loader, eps, pr_path)

    test_single(trained_model, test_loader, num_classes)


def get_loaders(train_data, test_data, batch_size):

    train_feats = torch.tensor([element[0] for element in train_data], dtype=torch.float32).requires_grad_(True)
    train_labels = torch.tensor([element[1] for element in train_data]).squeeze(1)
    train_ents = torch.LongTensor([element[2] for element in train_data]).squeeze(1)

    test_feats = torch.tensor([element[0] for element in test_data], dtype=torch.float32).requires_grad_(True)
    test_labels = torch.tensor([element[1] for element in test_data]).squeeze(1)
    test_ents = torch.LongTensor([element[2] for element in test_data]).squeeze(1)

    print('Shape of tensors train_joint :', 'Feats: ', train_feats.shape, 'Ents: ', train_ents.shape, 'Labels: ',
          train_labels.shape)
    print('Shape of tensors train_joint :', 'Feats: ', test_feats.shape, 'Ents: ', test_ents.shape, 'Labels: ',
          test_labels.shape)

    dataset_train = torch.utils.data.TensorDataset(train_feats, train_ents, train_labels)
    dataset_test = torch.utils.data.TensorDataset(test_feats, test_ents, test_labels)

    return torch.utils.data.DataLoader(dataset_train, batch_size=batch_size), \
           torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)


if __name__ == "__main__":

    confíg = config_loader.get_config(FEEDFORWARD_CONFIG, interpolation=True)
    learning_rate = confíg.getfloat("TRAINING", "learning_rate")
    epochs = confíg.getint("TRAINING", "epochs")
    gamma = confíg.getfloat("TRAINING", "gamma")
    batch_size = confíg.getint("GENERAL", "batch_size")
    project_dir_path = confíg.get("GENERAL", "project_dir")
    num_classes = confíg.getint("GENERAL", "num_classes")
    model_name = confíg.get("ARCHITECTURE", "model")
    dataset = confíg.get("DATA", "dataset")
    available_joint_models = json.loads(confíg.get("GENERAL", "implemented_joint_models"))
    available_single_models = json.loads(confíg.get("GENERAL", "implemented_single_models"))
    input_size = None
    hidden_size = None

    if confíg.has_option("ARCHITECTURE", "input_size"):
        input_size = confíg.getint("ARCHITECTURE", "input_size")

    if confíg.has_option("ARCHITECTURE", "hidden_size"):
        hidden_size = confíg.getint("ARCHITECTURE", "hidden_size")

    if len(sys.argv) > 1:
        epochs, gamma = sys.argv[1:]
        epochs, gamma = int(epochs[0]), int(gamma[0])

    if model_name not in available_single_models + available_joint_models:
        raise(NotImplementedError(f"Unknown Model {model_name}"))

    if model_name in available_joint_models:
        run_joint_model(model_name,
                        available_joint_models,
                        dataset, epochs, gamma, learning_rate, batch_size, project_dir_path)
    if model_name in available_single_models:
        run_single_model(model_name, available_single_models, input_size, hidden_size, num_classes,
                        dataset, epochs, learning_rate, batch_size, project_dir_path)