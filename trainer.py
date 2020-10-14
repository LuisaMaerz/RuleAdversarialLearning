import random
import numpy as np
import shutil
import torch
import logging
import torch.utils.data
import torch.nn as nn
import sys
import os
from dann3_model import DANN3
from data_handling.make_toy_data import make_combined_toy_dataset
from nlp_toolkit.utility import config_loader

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_FILE_LOCATION = os.path.abspath(os.path.dirname(__file__))
DANN3_CONFIG = CURRENT_FILE_LOCATION + "/config/dann3_experiment.cfg"


def run_DANN3(eps, g, lr, bs, pr_path):
    data_train, data_test = make_combined_toy_dataset()
    train_loader, test_loader = get_loaders(data_train, data_test, batch_size = bs)

    net = DANN3()
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trained_model = train(net, optimizer, criterion, criterion2, train_loader, eps, g, pr_path)

    test(trained_model, test_loader)


def test(model, test_loader):
    print('NOW TESTING')
    net = model
    net.eval()

    predictions = []
    batch_labels = []
    probas = []


    alpha = 0
    batch_count = 0

    for sent, ents, labels in test_loader:
        print('\nTestinstance 1: ', sent.data.tolist()[0],
              '\nTestinstance 2: ', sent.data.tolist()[1])

        batch_count += 1
        sent, ents, labels = sent.cuda(), ents.cuda(), labels.cuda()

        outputs, _ = net(sent, alpha=alpha)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)

        probabilities.cuda()
        probas.append(probabilities.data.tolist())

        _, predicted = torch.max(outputs.data, -1)

        predictions += predicted.tolist()
        batch_labels += labels.tolist()


    predictions = [str(p) for p in predictions]
    batch_labels = [str(p) for p in batch_labels]

    print('\nProbability of class 1 for Testinstance 1:', probas[0][0][0],
          '\nProbability of class 2 for Testinstance 1:', probas[0][0][1],
          '\n\nProbability of class 1 for Testinstance 2:', probas[0][1][0],
          '\nProbability of class 2 for Testinstance 2:', probas[0][1][1], '\n')
    #print('predicted: ', predictions , '\ngold_labels: ' , batch_labels)
    #p, r, f1 = metrics.score(batch_labels, predictions, verbose=True)


def create_summary_writer(use_tensorboard):
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            tensorboard_dir = os.path.join(project_dir_path, "tensorboard")
            if os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir, ignore_errors=False, onerror=None)
            writer = SummaryWriter(log_dir=tensorboard_dir, comment=project_dir_path.split("/")[-1])
            print(f"tensorboard logging path is {tensorboard_dir}")
        except:
            logging.warning(
                "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
            )
            use_tensorboard = False
            return None
    return writer


def train(net, optimizer, criterion, criterion2, train_loader, epochs, gamma, project_dir_path, use_tensorboard="True"):
    losses_dict = {}
    print('NOW TRAINING')
    counter = 0
    print_every = 10

    tensorboard_writer = create_summary_writer(use_tensorboard)

    len_dataloader = len(train_loader)
    for e in range(epochs):
        net.train()
        i = 1
        # batch loop
        for inputs, ents, labels in train_loader:
            counter += 1

            inputs, ents, labels = inputs.cuda(), ents.cuda(), labels.cuda()

            p = float(i + e * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # get the output from the model
            rel_pred_output, ent_pred_output = net(inputs, alpha=alpha)

            # calculate the loss and perform backprop
            # add losses to dict for logging
            rel_pred_error = criterion(rel_pred_output, labels)
            losses_dict["relation_pred_error"] = rel_pred_error
            ent_pred_error = criterion2(ent_pred_output, ents)
            losses_dict["entity_pred_error"] = ent_pred_error

            err = rel_pred_error + ent_pred_error * gamma
            err = ent_pred_error * gamma
            losses_dict["combined_error"] = err

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if use_tensorboard:
                _log_losses(tensorboard_writer, losses_dict, e)

                # print(weights_to_print)
                if e < 5 or e % 10 == 0: # print first epochs andb then every 10th epoch
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                                tensorboard_writer.add_histogram(name, param, e)
                    #if self.safe_checkpoint_regularly:
                        #self.save_checkpoint(base_path / f"model-epoch_{self.epoch}.pt")

            # print('\nWeights entity classifier:', net.ent_classifier.ent_classifier.weight,
            #       '\nWeights class classifier:', net.classifier.class_classifier.weight,
            #       '\nWeights feat:', net.feature.hidden.weight)

            # print('\nMultiplied Matrix "A" - class x feature:\n', torch.mm( net.classifier.class_classifier.weight, net.feature.hidden.weight),
            #       '\n\nMultiplied Matrix "B" - ents x feature:\n', torch.mm(net.ent_classifier.ent_classifier.weight, net.feature.hidden.weight))


            i += 1

            # loss stats
            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss combi: {:.6f}...".format(err.item()),
                      "Loss Entities: {:.6f}...".format(ent_pred_error.item()),
                      "Loss Relation Extraction: {:.6f}...".format(rel_pred_error.item()))
    return net


def _log_losses(writer, loss_dict, epoch):
    for k, v in loss_dict.items():
        writer.add_scalar(k, loss_dict[k], epoch)


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
    #FB: num classes is not used ?
    #num_classes = confíg.getint("GENERAL", "num_classes")
    learning_rate = confíg.getfloat("TRAINING", "learning_rate")
    epochs = confíg.getint("TRAINING", "epochs")
    gamma = confíg.getfloat("TRAINING", "gamma")
    batch_size = confíg.getint("GENERAL", "batch_size")
    project_dir_path = confíg.get("GENERAL", "project_dir")

    if len(sys.argv) > 1:
        epochs, gamma = sys.argv[1:]
        epochs, gamma = int(epochs[0]), int(gamma[0])
    run_DANN3(epochs, gamma, learning_rate, batch_size, project_dir_path)