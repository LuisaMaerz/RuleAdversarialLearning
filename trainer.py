import numpy as np
import shutil
import torch
import logging
import torch.utils.data
import torch.nn as nn


#TODO: Adapt prints in testing to 4 classes
def test_joint(model, test_loader):
    print('NOW TESTING')
    #TODO: I think you can remove this assignment and do eval directly
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

    # Iterate ove the classes
    print('\nProbability of class 1 for Testinstance 1:', probas[0][0][0],
          '\nProbability of class 2 for Testinstance 1:', probas[0][0][1],
          '\n\nProbability of class 1 for Testinstance 2:', probas[0][1][0],
          '\nProbability of class 2 for Testinstance 2:', probas[0][1][1], '\n')
    #print('predicted: ', predictions , '\ngold_labels: ' , batch_labels)
    #p, r, f1 = metrics.score(batch_labels, predictions, verbose=True)


def test_single(model, test_loader, num_classes):
    print('NOW TESTING SINGLE MODEL')
    model.eval()
    probas = []

    batch_count = 0

    for sent, ents, labels in test_loader:
        print('\nTestinstance 1: ', sent.data.tolist()[0],
              '\nTestinstance 2: ', sent.data.tolist()[1])

        batch_count += 1
        sent, ents, labels = sent.cuda(), ents.cuda(), labels.cuda()

        outputs = model(sent)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)

        probabilities.cuda()
        probas.append(probabilities.data.tolist())

        predicted = torch.max(outputs.data, -1)

    for c in range(num_classes):
        print(f'\nProbability of class {c} for Testinstance 1: {probas[0][0][c]}',
              f'\n\nProbability of class {c} for Testinstance 2: {probas[0][1][c]}\n')
        print(f'predicted: {predicted}, gold: {labels}')
    #p, r, f1 = metrics.score(batch_labels, predictions, verbose=True)


def create_summary_writer(use_tensorboard, project_dir_path):
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
            return None
    return writer


def train_joint(net, optimizer, criterion, criterion2, train_loader, epochs, gamma, project_dir_path, use_tensorboard="True"):
    losses_dict = {}
    print('NOW DOING JOINT TRAINING')
    counter = 0
    print_every = 10

    tensorboard_writer = create_summary_writer(use_tensorboard, project_dir_path)

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
            # TODO: Check what happens with alpha, is it used in the formward pass?
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

            # TODO: I think the line below is a bug. It makes the network forget the gradients so far.
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


def train_single(
        net,
        optimizer,
        criterion,
        train_loader,
        epochs,
        project_dir_path,
        use_labels=True,
        use_ents=False,
        use_tensorboard="True"):

    losses_dict = {}
    print('NOW DOING SINGLE TRAINING')
    counter = 0
    print_every = 10

    tensorboard_writer = create_summary_writer(use_tensorboard, project_dir_path)

    net.train()
    for e in range(epochs):
        i = 1
        # batch loop
        for inputs, ents, labels in train_loader:
            counter += 1

            inputs, ents, labels = inputs.cuda(), ents.cuda(), labels.cuda()

            pred_output = net(inputs)


            if use_labels and use_ents:
                raise(ValueError("Cannot train single model on labels and pattern indicator features"))

            if use_labels:
                pred_error = criterion(pred_output, labels)

            if use_ents:
                pred_error = criterion(pred_output, ents)

            losses_dict["prediction_error"] = pred_error

            # TODO: I think this is a bug. It makes the network forget the gradients so far.
            #optimizer.zero_grad()
            pred_error.backward()
            optimizer.step()

            if use_tensorboard:
                _log_losses(tensorboard_writer, losses_dict, e)

                # print(weights_to_print)
                if e < 5 or e % 10 == 0: # print first epochs andb then every 10th epoch
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            tensorboard_writer.add_histogram(name, param, e)
            i += 1

            # loss stats
            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(pred_error.item()))

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

    print('Shape of tensors for training :', 'Feats: ', train_feats.shape, 'Ents: ', train_ents.shape, 'Labels: ',
          train_labels.shape)
    print('Shape of tensors for training :', 'Feats: ', test_feats.shape, 'Ents: ', test_ents.shape, 'Labels: ',
          test_labels.shape)


    dataset_train = torch.utils.data.TensorDataset(train_feats, train_ents, train_labels)
    dataset_test = torch.utils.data.TensorDataset(test_feats, test_ents, test_labels)

    return torch.utils.data.DataLoader(dataset_train, batch_size=batch_size), \
           torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)