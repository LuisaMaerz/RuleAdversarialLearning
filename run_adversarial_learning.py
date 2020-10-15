import numpy as np
import shutil
import torch
import logging
import torch.utils.data
import torch.nn as nn
import sys
import os
from models.joint_model import DANN3
from data_handling.make_toy_data import make_combined_toy_dataset
from nlp_toolkit.utility import config_loader
from trainer import train, test, create_summary_writer

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
DANN3_CONFIG = CURRENT_FILE_LOCATION + "/config/dann3_combined_toy.cfg"


def train_joint_model(joint_model, eps, g, lr, bs, pr_path):
    data_train, data_test = make_combined_toy_dataset()
    train_loader, test_loader = get_loaders(data_train, data_test, batch_size = bs)

    if joint_model == "DANN3":
        net = DANN3()
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model = train(net, optimizer, criterion, criterion2, train_loader, eps, g, pr_path)

    test(trained_model, test_loader)


def get_loaders(train_data, test_data, batch_size):

    train_feats = torch.tensor([element[0] for element in train_data], dtype=torch.float32).requires_grad_(True)
    train_labels = torch.tensor([element[1] for element in train_data]).squeeze(1)
    train_ents = torch.LongTensor([element[2] for element in train_data]).squeeze(1)

    test_feats = torch.tensor([element[0] for element in test_data], dtype=torch.float32).requires_grad_(True)
    test_labels = torch.tensor([element[1] for element in test_data]).squeeze(1)
    test_ents = torch.LongTensor([element[2] for element in test_data]).squeeze(1)

    print('Shape of tensors train :', 'Feats: ', train_feats.shape, 'Ents: ', train_ents.shape, 'Labels: ',
          train_labels.shape)
    print('Shape of tensors train :', 'Feats: ', test_feats.shape, 'Ents: ', test_ents.shape, 'Labels: ',
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
    model = confíg.get("TRAINING", "model")

    if len(sys.argv) > 1:
        epochs, gamma = sys.argv[1:]
        epochs, gamma = int(epochs[0]), int(gamma[0])
    train_joint_model(model, epochs, gamma, learning_rate, batch_size, project_dir_path)