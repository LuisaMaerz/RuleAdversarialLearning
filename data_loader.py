import numpy as np

import torch
from collections import Counter
import pickle

import utils


def get_words_to_id_map(textlist, vocab_size, name):
    """ Creates mapping word -> id for the most frequent words in the vocabulary. Ids 0 and 1 are reserved for the
    padding symbol <PAD> and the unknown token <UNK>. vocab_size determines the overall vocabulary size (including <UNK>
    and <PAD>)"""
    assert (vocab_size >= 2)
    c = Counter(tok for text in textlist for tok in text.split(" "))
    try:
        # Use fixed word frequencies
        with open("../data/ids_words" + str(vocab_size) + ".p", "rb") as f:
            ids_words = pickle.load(f)
    except FileNotFoundError:
        ids_words = enumerate(['<PAD>', '<UNK>'] + sorted([word for word, count in c.most_common(vocab_size - 2)]))
        # Write file instead of extensive deterministic function
        with open("../data/" + name + str(vocab_size) + ".p", "wb") as f:
            pickle.dump(ids_words, f)
    return {w: idx for idx, w in ids_words}


def get_text_matrix(textlist, word_to_id, maxlen):
    """ This takes textlist (list with white-space separated tokens) and returns a numpy matrix of size
    len(textlist) x maxlen.
    Each row in the matrix contains for each text the sequence of word ids (i.e. the columns correspond to the positions
    in the sentence).
    If a sentence is longer than maxlen, it is truncated after maxlen tokens.
    If a sentence is shorter than maxlen, it is filled with 0 (= the word id of the <PAD> token).
    """
    m = np.zeros((len(textlist), maxlen), dtype=int)
    row_nr, col_nr = 0, 0
    for text in textlist:
        col_nr = 0
        for word in text.split(" "):
            if col_nr == maxlen:
                break
            m[row_nr, col_nr] = word_to_id.get(word, 1)  # id for <UNK>
            col_nr += 1
        row_nr += 1
    return m


def convert2tensor(words, types, labels, lens, ents, batch_size):
    """ Provides data in Pytorch Dataloader format.
    Everything is put on GPU.
    Lens are needed for later padding.
    """
    target = np.asarray(labels, dtype=np.long)
    tensor_target = torch.from_numpy(target).cuda()
    lens = torch.IntTensor(lens).cuda()

    tensor_words = torch.LongTensor(words).cuda()
    tensor_types = torch.LongTensor(types).cuda()
    tensor_ents = torch.LongTensor(ents).cuda()

    print('Shape of tensors :', 'Data: ', tensor_words.shape, 'Types: ', tensor_types.shape, 'Labels: ',
          tensor_target.shape, 'Entities: ', tensor_ents.shape, 'Lens: ', lens.shape)
    dataset = torch.utils.data.TensorDataset(tensor_words, tensor_types, tensor_target, lens, tensor_ents)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def get_loaders(word_emb_file, train_file, dev_file, test_file, num_ents, clipping = None, no_rel = None):
    """ Prepares data and returns Pytorch data_loaders.
    It takes three filenames as arguments as well as the file with the pre-trained word embeddings.
    The script assumes that all relation types occur at least once in the training data.

    Data is first split in tokens and types of the entities.
    Data is truncated to max_len and padded if necessary.
    Labels are represented as categorical single numbers.
    """
    batch_size = 32

    train_tokens, train_types, train_labels = utils.middletokens_types_labels(train_file)
    dev_tokens, dev_types, dev_labels = utils.middletokens_types_labels(dev_file)
    test_tokens, test_types, test_labels = utils.middletokens_types_labels(test_file)

    train_ents, train_dict = utils.get_entity_set(train_file)
    dev_ents, _ = utils.get_entity_set(dev_file)
    test_ents, _ = utils.get_entity_set(test_file)

    ent2idx = utils.get_ent_voc(train_dict, num_ents)

    train_ents_idx = [ent2idx.get(entity, 0) for entity in train_ents]
    dev_ents_idx = [ent2idx.get(entity, 0) for entity in dev_ents]
    test_ents_idx = [ent2idx.get(entity, 0) for entity in test_ents]

    if no_rel != None:
        train_tokens, train_types, train_ents_idx, train_labels = no_rel_filter(train_tokens, train_types, train_ents_idx, train_labels)
        dev_tokens, dev_types, dev_ents_idx, dev_labels = no_rel_filter(dev_tokens, dev_types, dev_ents_idx, dev_labels)
        test_tokens, test_types, test_ents_idx, test_labels = no_rel_filter(test_tokens, test_types, test_ents_idx, test_labels)

    ###################################################################################################################
    ################## clipped data set with only top n entities, equal number of sentences of each entity
    if clipping != None:
        cut_off = len([x for x in train_ents if str(x) == list(ent2idx.keys())[list(ent2idx.values()).index(num_ents-1)]]) #freq of last topn item
        train_tokens, train_types, train_labels, train_ents_idx = create_topn_set(train_tokens, train_types, train_labels, train_ents_idx, cut_off)


    train_lens = [len(sent.split(' ')) if len(sent.split(' ')) <= 50 else 50 for sent in train_tokens]
    dev_lens = [len(sent.split(' ')) if len(sent.split(' ')) <= 50 else 50 for sent in dev_tokens]
    test_lens = [len(sent.split(' ')) if len(sent.split(' ')) <= 50 else 50 for sent in test_tokens]

    labels2idx = utils.get_label_dict(train_labels + dev_labels + test_labels)
    idx2labels = dict([(value, key) for key, value in labels2idx.items()])

    train_labels = [labels2idx[label] for label in train_labels]
    dev_labels = [labels2idx[label] for label in dev_labels]
    test_labels = [labels2idx[label] for label in test_labels]

    # Convert text and labels to matrix format.
    word_to_id, word_embedding_matrix = utils.vocab_and_vectors(word_emb_file, ['<PAD>', '<UNK>'])

    num_types = 20
    type_to_id = get_words_to_id_map(train_types, num_types, "types")

    maxlen = 50

    train_word_matrix = get_text_matrix(train_tokens, word_to_id, maxlen)
    train_type_matrix = get_text_matrix(train_types, type_to_id, 2)

    dev_word_matrix = get_text_matrix(dev_tokens, word_to_id, maxlen)
    dev_type_matrix = get_text_matrix(dev_types, type_to_id, 2)

    test_word_matrix = get_text_matrix(test_tokens, word_to_id, maxlen)
    test_type_matrix = get_text_matrix(test_types, type_to_id, 2)

    mask_matrix = utils.get_mapping_ent2rel(train_ents_idx + dev_ents_idx + test_ents_idx,
                                            train_labels + dev_labels + test_labels, len(labels2idx), num_ents)


    word_input_dim = word_embedding_matrix.shape[0]
    word_output_dim = word_embedding_matrix.shape[1]

    train_loader = convert2tensor(train_word_matrix, train_type_matrix, train_labels, train_lens, train_ents_idx, batch_size)
    valid_loader = convert2tensor(dev_word_matrix, dev_type_matrix, dev_labels, dev_lens, dev_ents_idx, batch_size)
    test_loader = convert2tensor(test_word_matrix, test_type_matrix, test_labels, test_lens, test_ents_idx, batch_size)

    return train_loader, valid_loader, test_loader, word_input_dim, word_output_dim, word_embedding_matrix, idx2labels, mask_matrix



def create_topn_set(tokens, types, labels, ents, cut_off):
    print('Instances per entity: ', cut_off)
    clipped_tokens = []
    clipped_types = []
    clipped_labels = []
    clipped_ents = []
    for i in range(len(tokens)):
        if ents[i] != 0:
            if (len([x for x in clipped_ents if x == ents[i]])) < cut_off:
                clipped_tokens.append(tokens[i])
                clipped_types.append(types[i])
                clipped_labels.append(labels[i])
                clipped_ents.append(ents[i])
    return clipped_tokens, clipped_types, clipped_labels, clipped_ents


def no_rel_filter(tokens, types, ents, labels):
    new_tokens = [tokens[i] for i in range(len(tokens)) if str(labels[i]) != 'no_relation']
    new_ents = [ents[i] for i in range(len(tokens)) if str(labels[i]) != 'no_relation']
    new_types = [types[i] for i in range(len(tokens)) if str(labels[i]) != 'no_relation']
    new_labels = [labels[i] for i in range(len(tokens)) if str(labels[i]) != 'no_relation']
    return new_tokens, new_types, new_ents, new_labels

