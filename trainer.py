import numpy as np
import shutil
import torch
import logging
import torch.utils.data
import torch.nn as nn


def test_joint(model, test_loader, task_type, device):
    print('NOW TESTING')
    model.eval()

    predictions = []
    batch_labels = []
    probas = []

    alpha = 0
    batch_count = 0

    for sent, ents, labels in test_loader:
        print('\nTestinstance 1: ', sent.data.tolist()[0],
              '\nTestinstance 2: ', sent.data.tolist()[1])

        if task_type != 'binary':
            print('\nTestinstance 3: ', sent.data.tolist()[2],
                  '\nTestinstance 4: ', sent.data.tolist()[3])

        batch_count += 1

        with torch.cuda.device(device):
            sent, ents, labels = sent.cuda(), ents.cuda(), labels.cuda()

        outputs, _ = model(sent, alpha=alpha)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)

        with torch.cuda.device(device):
            probabilities.cuda()
        probas.append(probabilities.data.tolist())

        _, predicted = torch.max(outputs.data, -1)

        predictions += predicted.tolist()
        batch_labels += labels.tolist()

    predictions = [str(p) for p in predictions]
    batch_labels = [str(p) for p in batch_labels]

    # Iterate ove the classes
    if task_type == "binary":
        print('\nProbability of class 1 for Testinstance 1:', probas[0][0][0],
              '\nProbability of class 2 for Testinstance 1:', probas[0][0][1],
              '\nProbability of class 1 for Testinstance 2:', probas[0][1][0],
              '\nProbability of class 2 for Testinstance 2:', probas[0][1][1], '\n')
    else:
        print('\nProbability of class 1 for Testinstance 1:', probas[0][0][0],
              '\nProbability of class 2 for Testinstance 1:', probas[0][0][1],
              '\n\nProbability of class 3 for Testinstance 1:', probas[0][0][2],
              '\nProbability of class 4 for Testinstance 1:', probas[0][0][3], '\n')
        print('\nProbability of class 1 for Testinstance 2:', probas[0][1][0],
              '\nProbability of class 2 for Testinstance 2:', probas[0][1][1],
              '\nProbability of class 3 for Testinstance 2:', probas[0][1][2],
              '\nProbability of class 4 for Testinstance 2:', probas[0][1][3], '\n')
        print('\nProbability of class 1 for Testinstance 3:', probas[0][2][0],
              '\nProbability of class 2 for Testinstance 3:', probas[0][2][1],
              '\nProbability of class 3 for Testinstance 3:', probas[0][2][2],
              '\nProbability of class 4 for Testinstance 3:', probas[0][2][3], '\n')
        print('\nProbability of class 1 for Testinstance 4:', probas[0][3][0],
              '\nProbability of class 2 for Testinstance 4:', probas[0][3][1],
              '\nProbability of class 3 for Testinstance 4:', probas[0][3][2],
              '\nProbability of class 4 for Testinstance 4:', probas[0][3][3], '\n')

    print('predicted: ', predictions , '\ngold_labels: ' , batch_labels)


def test_single(model, test_loader, num_classes, binary, device):
    print('NOW TESTING SINGLE MODEL')

    model.eval()
    probas = []

    batch_count = 0

    for sent, ents, labels in test_loader:
        print('\nTestinstance 1: ', sent.data.tolist()[0],
              '\nTestinstance 2: ', sent.data.tolist()[1])

        if binary != 'binary':
            print('\nTestinstance 3: ', sent.data.tolist()[2],
                  '\nTestinstance 4: ', sent.data.tolist()[3])

        batch_count += 1

        with torch.cuda.device(device):
            sent, ents, labels = sent.cuda(), ents.cuda(), labels.cuda()
        outputs = model(sent)

        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)

        with torch.cuda.device(device):
            probabilities.cuda()
        probas.append(probabilities.data.tolist())

        predicted = torch.max(outputs.data, -1)

    if binary == 'binary':
        for c in range(num_classes):
            print(f'\nProbability of class {c} for Testinstance 1: {probas[0][0][c]}',
                  f'\n\nProbability of class {c} for Testinstance 2: {probas[0][1][c]}\n')
            print(f'predicted: {predicted}, gold: {labels}')
    else:
        for c in range(num_classes):
            print(f'\nProbability of class {c} for Testinstance 1: {probas[0][0][c]}',
                  f'\n\nProbability of class {c} for Testinstance 2: {probas[0][1][c]}\n',
                  f'\n\nProbability of class {c} for Testinstance 3: {probas[0][2][c]}\n',
                  f'\n\nProbability of class {c} for Testinstance 4: {probas[0][3][c]}\n')
            print(f'predicted: {predicted}, gold: {labels}')


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


def train_joint(net, optimizer, criterion, criterion2, train_loader, epochs, gamma, project_dir_path, device, use_tensorboard="True"):
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

            with torch.cuda.device(device):
                inputs, ents, labels = inputs.cuda(), ents.cuda(), labels.cuda()

            p = float(i + e * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # show the complete input to the model but only compute loss on the labels
            rel_pred_output, ent_pred_output = net(inputs, alpha=alpha)

            # calculate the loss and perform backprop
            # add losses to dict for logging
            rel_pred_error = criterion(rel_pred_output, labels)
            losses_dict["relation_pred_error"] = rel_pred_error
            ent_pred_error = criterion2(ent_pred_output, ents)
            losses_dict["entity_pred_error"] = ent_pred_error

            err = rel_pred_error + ent_pred_error * gamma
            losses_dict["combined_error"] = err

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if use_tensorboard:
                _log_losses(tensorboard_writer, losses_dict, e)

                # print(weights_to_print)
                if e < 5 or e % 10 == 0: # print first epochs and then every 10th epoch
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
        use_labels,
        use_ents,
        device,
        use_tensorboard="True"):

    losses_dict = {}
    print('NOW DOING SINGLE TRAINING')
    counter = 0
    print_every = 10

    tensorboard_writer = create_summary_writer(use_tensorboard, project_dir_path)

    print(torch.cuda.current_device())
    net.train()
    for e in range(epochs):
        i = 1
        # batch loop
        for inputs, ents, labels in train_loader:
            counter += 1

            with torch.cuda.device(device):
                inputs, ents, labels = inputs.cuda(), ents.cuda(), labels.cuda()

            pred_output = net(inputs)

            if use_labels == 'True' and use_ents == 'True':
                raise(ValueError("Cannot train single model on labels and pattern indicator features"))

            if use_labels == 'True':
                pred_error = criterion(pred_output, labels)

            if use_ents == 'True':
                pred_error = criterion(pred_output, ents)

            losses_dict["prediction_error"] = pred_error

            optimizer.zero_grad()
            pred_error.backward()
            optimizer.step()

            if use_tensorboard:
                _log_losses(tensorboard_writer, losses_dict, e)

                # print(weights_to_print)
                if e < 5 or e % 10 == 0: # print first epochs and then every 10th epoch
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

