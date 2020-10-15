import numpy as np
import shutil
import torch
import logging
import torch.utils.data
import torch.nn as nn
import sys
import os
from models.joint_model import DANN3
from data_handling.make_toy_data import make_combined_toy_dataset, make_pattern_toy_dataset, make_class_features_toy_dataset
from nlp_toolkit.utility import config_loader
from trainer import train_joint, test, create_summary_writer

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
DANN3_CONFIG = CURRENT_FILE_LOCATION + "/config/dann3_combined_toy.cfg"


def laod_dataset_name(data_set_name, available_data_sets):
    if data_set_name not in available_data_sets:
        raise(FileNotFoundError(f"Dataset {data_set_name} does not exist"))

    if data_set_name == "combined_toy":
        data_train, data_test = make_combined_toy_dataset()

    if data_set_name == "rule_patterns_toy":
        data_train, data_test = make_pattern_toy_dataset()

    if data_set_name == "class_features_toy":
        data_train, data_test = make_class_features_toy_dataset()

    return data_train, data_test


def run_joint_model(model, available_joint_models, data_set_name, eps, g, lr, bs, pr_path):

    if model not in available_joint_models:
        raise(NotImplementedError(f"Model {model} not implemented"))

    if model == "DANN3":
        net = DANN3()
    net.cuda()

    data_train, data_test = laod_dataset_name(data_set_name)
    train_loader, test_loader = get_loaders(data_train, data_test, batch_size = bs)

    classifer_1_loss = nn.CrossEntropyLoss()
    classifier_2_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model = train_joint(net, optimizer, classifer_1_loss, classifier_2_loss, train_loader, eps, g, pr_path)

    test(trained_model, test_loader)


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

    confíg = config_loader.get_config(DANN3_CONFIG, interpolation=True)
    learning_rate = confíg.getfloat("TRAINING", "learning_rate")
    epochs = confíg.getint("TRAINING", "epochs")
    gamma = confíg.getfloat("TRAINING", "gamma")
    batch_size = confíg.getint("GENERAL", "batch_size")
    project_dir_path = confíg.get("GENERAL", "project_dir")
    model_name = confíg.get("ARCHITECTURE", "model")
    dataset = confíg.get("DATA", "dataset")

    if len(sys.argv) > 1:
        epochs, gamma = sys.argv[1:]
        epochs, gamma = int(epochs[0]), int(gamma[0])
    run_joint_model(model_name, dataset, epochs, gamma, learning_rate, batch_size, project_dir_path)