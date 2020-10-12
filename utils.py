import numpy as np
from collections import defaultdict
import operator
import torch


def tokens_and_labels(filename):
    """ This reads a file with relation/sentence instances (sentences
    in CONLL format) and returns two lists:
    textlist: list of strings; The white-space-seperated tokens of each
        instance, where tokens corresponding to subject are replaced by
        "<SUBJ"+subj_type+">" (subj_type is the named entity type of the
        subject).
        Tokens corresponding to object are treated analogously.
    labellist: list of strings; The relation names for all instances.
    """
    textlist = []
    labellist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# id="):
                # Instance starts
                parts = line.split(" ")
                rel_name = parts[3][5:]
                subj_type = ""
                obj_type = ""
                tokens = []
            elif line == "":
                # Instance ends
                labellist.append(rel_name)
                textlist.append(" ".join(tokens))
            elif line.startswith("#"):
                # comment
                pass
            else:
                parts = line.split("\t")
                token = parts[1]
                if parts[2] == "SUBJECT":
                    if subj_type == "":
                        # Subj not yet processed.
                        subj_type = parts[3]
                        tokens.append("<SUBJ"+subj_type+">")
                elif parts[4] == "OBJECT":
                    if obj_type == "":
                        obj_type = parts[5]
                        tokens.append("<OBJ"+obj_type+">")
                else:
                    tokens.append(token)
    return textlist, labellist


def middletokens_types_labels(filename):
    """ This reads a file with relation/sentence instances (sentences
    in CONLL format) and returns two lists:
    textlist: list of strings; The white-space-seperated tokens of each
        instance, where tokens corresponding to subject are replaced by
        "<SUBJ"+subj_type+">" (subj_type is the named entity type of the
        subject).
        Tokens corresponding to object are treated analogously.
    labellist: list of strings; The relation names for all instances.
    """
    textlist = []
    typeslist = []
    labellist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# id="):
                # Instance starts
                parts = line.split(" ")
                rel_name = parts[3][5:]
                subj_type = ""
                obj_type = ""
                tokens = []
                types = []
            elif line == "":
                # Instance ends
                labellist.append(rel_name)
                textlist.append(" ".join(tokens))
                typeslist.append(" ".join(types))
                #typeslist.append(subj_type + " " + obj_type)
            elif line.startswith("#"):
                # comment
                pass
            else:
                parts = line.split("\t")
                token = parts[1]
                if parts[2] == "SUBJECT":
                    if subj_type == "":
                        # Subj not yet processed.
                        subj_type = parts[3]
                        types.append("SUBJ_" + subj_type)
                    tokens.append(token)
                elif parts[4] == "OBJECT":
                    if obj_type == "":
                        obj_type = parts[5]
                        types.append("OBJ_" + obj_type)
                    tokens.append(token)
                elif (subj_type == "" and obj_type != "") or (subj_type != "" and obj_type == ""):
                    # Only middle tokens
                    tokens.append(token)
    return textlist, typeslist, labellist


def middletokens_types_labels_window(filename):
    """ This reads a file with relation/sentence instances (sentences
    in CONLL format) and returns two lists:
    textlist: list of strings; The white-space-seperated tokens of each
        instance, where tokens corresponding to subject are replaced by
        "<SUBJ"+subj_type+">" (subj_type is the named entity type of the
        subject).
        Tokens corresponding to object are treated analogously.
    labellist: list of strings; The relation names for all instances.
    """
    textlist = []
    typeslist = []
    labellist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# id="):
                # Instance starts
                parts = line.split(" ")
                rel_name = parts[3][5:]
                subj_type = ""
                obj_type = ""
                token = ""
                prev_token = ""
                tokens = []
                subj_obj = False
                prev_subj_obj = False
            elif line == "":
                # Instance ends
                labellist.append(rel_name)
                textlist.append(" ".join(tokens))
                typeslist.append(subj_type + " " + obj_type)
            elif line.startswith("#"):
                # comment
                pass
            else:
                parts = line.split("\t")
                prev_token = token
                prev_subj_obj = subj_obj
                token = parts[1]
                subj_obj = False
                if parts[2] == "SUBJECT":
                    if subj_type == "":
                        # Subj not yet processed.
                        subj_type = parts[3]
                        if obj_type == "":
                            tokens.append(prev_token)
                    subj_obj = True
                    tokens.append(token)
                elif parts[4] == "OBJECT":
                    if obj_type == "":
                        obj_type = parts[5]
                        if subj_type == "":
                            tokens.append(prev_token)
                    subj_obj = True
                    tokens.append(token)
                elif (subj_type == "" and obj_type != "") or (subj_type != "" and obj_type == ""):
                    # Only middle tokens
                    tokens.append(token)
                elif subj_type != "" and obj_type != "" and prev_subj_obj == True: # just processed last subj/obj
                    tokens.append(token)

    return textlist, typeslist, labellist


def vocab_and_vectors(filename, special_tokens):
    """special tokens have all-zero word vectors"""
    with open(filename) as in_file:
        parts = in_file.readline().strip().split(" ")
        num_vecs = int(parts[0]) + len(special_tokens)
        dim = int(parts[1])

        matrix = np.zeros((num_vecs, dim))
        word_to_id = dict()

        nextword_id = len(special_tokens)
        for line in in_file:
            parts = line.strip().split(' ')
            word = parts[0]
            emb = [float(v) for v in parts[1:]]
            matrix[nextword_id] = emb
            word_to_id[word] = nextword_id
            nextword_id += 1

    return word_to_id, matrix


def get_label_dict(labels):
    labels = set(labels)
    if 'no_relation' in labels:
        labels.remove('no_relation')
    label2idx = {}
    label2idx['no_relation'] = 0
    count = 1
    for label in labels:
        label2idx[label] = count
        count += 1

    return label2idx


def get_predef_label_dict(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    label2idx = {}
    count = 0
    for line in lines:
        label = line.strip().split('\t')[0]
        label2idx[label] = count
        count += 1
    return label2idx


def get_entity_set(path_dataset):
    """
    Load dataset into memory from text file
    Creates list of entity pairs for each sentence
    Creates dict with frequency of every entity pair
    """
    e1, e2 = "", ""
    entity_dict= defaultdict(int)
    entity_list = []

    with open(path_dataset, 'r') as f:
        for line in f:
            if line == '\n':
                entity_dict[(e1 +': '+ e2)[:-1]] += 1
                entity_list.append((e1 +': '+ e2)[:-1])
                e1, e2 = "", ""
            else:
                if len(line.split('\t')) > 1:
                    parts = line.strip().split('\t')
                    if parts[2] == 'SUBJECT':
                        e1 += parts[1] + ' '
                    elif parts[4] == 'OBJECT':
                        e2 += parts[1] + ' '
    sorted_dict = dict( sorted(entity_dict.items(), key=operator.itemgetter(1),reverse=True))
    return entity_list, sorted_dict


def get_ent_voc(ent_dict, top_n):
    '''
    :param ent_dict: dictionary with entity pairs and freqs
    :param top_n: how many of the most frequent entity pairs should be used
    :return: dictionary with entiy pairs and ids starting from 1 to reserve 0 as value for other entity pairs
    '''
    ent_ids = {}
    most_freq = [x for x in list(ent_dict)[:top_n-1]] #-1 because of indexing
    count = 1
    for entity in most_freq:
        ent_ids[entity] = count
        count += 1
    return ent_ids


def get_mapping_ent2rel(sent_ents, sent_rels, num_classes, top_n):
    '''
    :param sent_ents: entity pair id for each sentence
    :param sent_rels: label id for each sentence
    :return: matrix, rows correspond to relations, columns to entity pairs
    sets 1. if an entity pair occurs with a label and 0 if not
    '''
    print('Start to create mapping')

    mask_matrix = torch.tensor((), dtype=torch.float64)
    mask_matrix = mask_matrix.new_full((num_classes, top_n), 0)

    for i in range(len(sent_ents)):
        mask_matrix[sent_rels[i]][sent_ents[i]] = 1.

    print('Finished mapping')

    return mask_matrix


def get_label_mapping(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    count = 0
    label2idx = {}
    label2mapping = {}
    for line in lines:
        label = line.strip().split('\t')[0]
        mapping = line.strip().split('\t')[4]
        label2mapping[label] = mapping
        if mapping != 'XXXX' and not mapping in label2idx.keys():
            label2idx[mapping] = count
            count += 1

    return label2mapping, label2idx


def filter_data(tokens, tags, labels, mapping):
    '''
    filters data w.r.t. labels, to fit with other datasets
    '''
    new_tokens, new_tags, new_labels = [],[],[]
    for i,label in enumerate(labels):
        if mapping[label] != 'XXXX':
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])
            new_labels.append(mapping[label])

    return new_tokens, new_tags, new_labels

