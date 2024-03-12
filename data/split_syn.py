"""
Split up the training set and the test set of SynWoodScape.

"""

import random

with open('data/all_syn.txt', 'r') as file:
    lines = file.readlines()

random.seed(42)

index_list = list(range(len(lines)))

random.shuffle(index_list)

split_index = int(0.8 * len(index_list))

train_indexes = index_list[:split_index]
test_indexes = index_list[split_index:]

train_file_names = [lines[idx] for idx in sorted(train_indexes)]
test_file_names = [lines[idx] for idx in sorted(test_indexes)]

with open('train_syn.txt', 'w') as train_file:
    train_file.writelines(train_file_names)

with open('test_syn.txt', 'w') as test_file:
    test_file.writelines(test_file_names)
