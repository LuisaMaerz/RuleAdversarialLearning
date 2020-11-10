import torch


def make_data_loaders(train_data, test_data, batch_size, use_classes=False, use_ents=False):

    if use_classes:
        train_feats = torch.tensor([element[0][4:] for element in train_data], dtype=torch.float32).requires_grad_(True)
        test_feats = torch.tensor([element[0][4:] for element in test_data], dtype=torch.float32).requires_grad_(True)

    elif use_ents:
        train_feats = torch.tensor([element[0][:4] for element in train_data], dtype=torch.float32).requires_grad_(True)
        test_feats = torch.tensor([element[0][:4] for element in test_data], dtype=torch.float32).requires_grad_(True)

    else:
        train_feats = torch.tensor([element[0] for element in train_data], dtype=torch.float32).requires_grad_(True)
        test_feats = torch.tensor([element[0] for element in test_data], dtype=torch.float32).requires_grad_(True)

    train_labels = torch.tensor([element[1] for element in train_data]).squeeze(1)
    train_ents = torch.LongTensor([element[2] for element in train_data]).squeeze(1)

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