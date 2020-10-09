import random
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import sys
import data_loader
from unit_test_model import DANN, DANN2, DANN3
#import metrics

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(argv):
    epochs, gamma = argv
    epochs, gamma = int(epochs[0]), int(gamma[0])

    num_classes = 2

    data_train, data_test = make_dataset()
    train_loader, test_loader = get_loaders(data_train, data_test)

    net = DANN3()
    net.cuda()

    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    trained_model = train(net, optimizer, criterion, criterion2, train_loader, epochs, gamma)

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
        #print(probabilities)

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


def train(net, optimizer, criterion, criterion2, train_loader, epochs, gamma):
    print('NOW TRAINING')
    counter = 0
    print_every = 10

    len_dataloader = len(train_loader)
    for e in range(epochs):
        net.train()
        i = 1
        # batch loop
        for inputs, ents, labels in train_loader:
            counter += 1

            inputs, ents, labels = inputs.cuda(), ents.cuda(), labels.cuda()

            # zero accumulated gradients
            #print('before step \n', net.domain_classifier.ent_classifier.weight, net.classifier.class_classifier.weight)


            p = float(i + e * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # get the output from the model
            rel_pred_output, ent_pred_output = net(inputs, alpha=alpha)

            # calculate the loss and perform backprop
            rel_pred_error = criterion(rel_pred_output, labels)
            ent_pred_error = criterion2(ent_pred_output, ents)

            err = rel_pred_error + ent_pred_error * gamma

            optimizer.zero_grad()
            err.backward()
            optimizer.step()


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




def make_dataset():

    examples_train = [([1, 0, 0, 0, 1, 0], [0], [1]),               # PF1, CF1 -> C1
                      ([0, 1, 0, 0, 0, 1], [1], [2]),               # PF2, CF2 -> C2
                      ([0, 0, 1, 0, 1, 0], [0], [3]),               # PF3, CF1 -> C1
                      ([0, 0, 0, 1, 0, 1], [1], [4]),               # PF4, CF2 -> C2
                      ([0, 0, 0, 0, 1, 0], [1], [0]),               # CF1 -> C2 (!)
                      ([0, 0, 0, 0, 0, 1], [0], [0])]               # CF2 -> C1 (!)


    # examples_train = [([0, 0, 0, 0, 1, 0], [0], [0]),               # PF1, CF1 -> C1
    #                   ([0, 0, 0, 0, 0, 1], [1], [0]),               # PF2, CF2 -> C2
    #                   ([0, 0, 0, 0, 1, 0], [0], [0]),               # PF3, CF1 -> C1
    #                   ([0, 0, 0, 0, 0, 1], [1], [0]),               # PF4, CF2 -> C2
    #                   ([0, 0, 0, 0, 1, 0], [1], [0]),               # CF1 -> C2 (!)
    #                   ([0, 0, 0, 0, 0, 1], [0], [0])]               # CF2 -> C1 (!)


    examples_test = [([0, 0, 0, 0, 1, 0], [0], [0]),                # CF1 -> P(C1)?
                     ([0, 0, 0, 0, 0, 1], [1], [0])]                # CF2 -> P(C2)?
                                                                    # ent_types hier vernachlässigbar




    dataset_train = [element for i in range(1000) for element in examples_train]
    random.shuffle(dataset_train)

    return dataset_train, examples_test


def get_loaders(train_data, test_data):

    batch_size = 64

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
    main(sys.argv[1:])

