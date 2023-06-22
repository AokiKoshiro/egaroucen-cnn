import numpy as np
from utils import print_board

np_train_dataset = np.load("./data/train_loser_othello_dataset.npy")
for i in range(17):
    print_board(np_train_dataset[i][:-1])
    print(np_train_dataset[i][-1])
print(np_train_dataset.shape)