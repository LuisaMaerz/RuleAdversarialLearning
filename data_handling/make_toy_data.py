import random


def make_multiclass_toy_dataset():

    # Last number: recap of pattern (4 first bits)
    # Class features: 4 last in pattern
    # Rule features: 4 fiest in pattern
    # Second column: labels

    # Wahrscheinlichkeit C0/1/2/3, wenn nur class features berücksichtigt sind: 2/12 = 0,16666
    # Wahrscheinichkeit C0/1/2/3, wenn alles berücksichtigt wird: 3/12 = 0,25

    examples_train = [([1, 0, 0, 0, 1, 0, 0, 0], [0], [1]),               # PF1, CF0 -> C0,
                      ([0, 1, 0, 0, 0, 1, 0, 0], [1], [2]),               # PF2, CF1 -> C1,
                      ([0, 0, 1, 0, 1, 0, 0, 0], [0], [3]),               # PF3, CF0 -> C0,
                      ([0, 0, 0, 1, 0, 1, 0, 0], [1], [4]),               # PF4, CF1 -> C1,
                      ([1, 0, 0, 0, 0, 0, 1, 0], [2], [1]),               # PF1, CF2 -> C2,
                      ([0, 1, 0, 0, 0, 0, 0, 1], [3], [2]),               # PF2, CF3 -> C3
                      ([0, 0, 1, 0, 0, 0, 1, 0], [2], [3]),               # PF3, CF2 -> C2,
                      ([0, 0, 0, 1, 0, 0, 0, 1], [3], [4]),               # PF4, CF3 -> C3,
                      ([0, 0, 0, 0, 1, 0, 0, 0], [1], [0]),               # CF0 -> C1 (!)
                      ([0, 0, 0, 0, 0, 1, 0, 0], [0], [0]),               # CF1 -> C0 (!)
                      ([0, 0, 0, 0, 0, 0, 0, 1], [2], [0]),               # CF2 -> C3 (!)
                      ([0, 0, 0, 0, 0, 0, 1, 0], [3], [0])]               # CF3 -> C2 (!)

    # FB: These patterns have never been seen
    examples_test = [([0, 0, 0, 0, 1, 0, 0, 0], [0], [0]),                # CF0 -> P(C0)? -> 2/12
                     ([0, 0, 0, 0, 0, 1, 0, 0], [1], [0]),                # CF1 -> P(C1)? -> 2/12
                     ([0, 0, 0, 0, 0, 0, 1, 0], [2], [0]),                # CF2 -> P(C2)? -> 2/12
                     ([0, 0, 0, 0, 0, 0, 0, 1], [3], [0])]                # CF3 -> P(C3)? -> 2/12
                                                                    # ent_types hier vernachlässigbar

    # copy as many fake examples as you want
    dataset_train = [element for i in range(1000) for element in examples_train]
    random.shuffle(dataset_train)

    return dataset_train, examples_test


def make_binary_toy_dataset():

    examples_train = [([1, 0, 0, 0, 1, 0], [0], [1]),  # PF1, CF1 -> C1
                      ([0, 1, 0, 0, 0, 1], [1], [2]),  # PF2, CF2 -> C2
                      ([0, 0, 1, 0, 1, 0], [0], [3]),  # PF3, CF1 -> C1
                      ([0, 0, 0, 1, 0, 1], [1], [4]),  # PF4, CF2 -> C2
                      ([0, 0, 0, 0, 1, 0], [1], [0]),  # CF1 -> C2 (!)
                      ([0, 0, 0, 0, 0, 1], [0], [0])]  # CF2 -> C1 (!)

    examples_test = [([0, 0, 0, 0, 1, 0], [0], [0]),  # CF1 -> P(C1)?
                     ([0, 0, 0, 0, 0, 1], [1], [0])]  # CF2 -> P(C2)?

    dataset_train = [element for i in range(1000) for element in examples_train]
    random.shuffle(dataset_train)

    return dataset_train, examples_test


