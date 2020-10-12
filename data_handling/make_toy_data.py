import random


def make_toy_dataset():

    examples_train = [([1, 0, 0, 0, 1, 0], [0], [1]),               # PF1, CF1 -> C1, CFC1: 0
                      ([0, 1, 0, 0, 0, 1], [1], [2]),               # PF2, CF2 -> C2, CFC2: 1
                      ([0, 0, 1, 0, 1, 0], [0], [3]),               # PF3, CF1 -> C1, CFC3: 0
                      ([0, 0, 0, 1, 0, 1], [1], [4]),               # PF4, CF2 -> C2, CFC4: 1
                      ([0, 0, 0, 0, 1, 0], [1], [0]),               # CF1 -> C2 (!)
                      ([0, 0, 0, 0, 0, 1], [0], [0])]               # CF2 -> C1 (!)

    # FB: These patterns have never been seen
    examples_test = [([0, 0, 0, 0, 1, 0], [0], [0]),                # CF1 -> P(C1)? -> 1/3
                     ([0, 0, 0, 0, 0, 1], [1], [0])]                # CF2 -> P(C2)? -> 1/3
                                                                    # ent_types hier vernachlässigbar

    # copy as many fake examples as you want
    dataset_train = [element for i in range(100) for element in examples_train]
    random.shuffle(dataset_train)

    return dataset_train, examples_test