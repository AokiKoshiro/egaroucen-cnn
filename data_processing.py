import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import transcript2moves, moves2matrices, get_boards_loser_indices


def prepare_othello_dataset():
    transcripts = []
    for file_path in glob.glob('data/original_data/**/*.txt'):
        with open(file_path, 'r') as file:
            for transcript in file.readlines():
                transcripts.append(transcript.rstrip())
    print("Number of matches:", len(transcripts))
    df = pd.DataFrame()
    df["transcript"] = pd.DataFrame(transcripts)
    df["move"] = df["transcript"].apply(transcript2moves)
    df["matrix"] = df["move"].apply(moves2matrices)
    df[["board", "loser_index"]] = df.apply(lambda x: get_boards_loser_indices(x["move"]), axis=1, result_type="expand")
    df["board"] = df["board"].apply(lambda x: x[:-1])
    np_loser_move_matrix = create_loser_move_matrix(df["matrix"], df["loser_index"])
    np_loser_board = create_loser_board(df["board"], df["loser_index"])
    np_dataset = np.concatenate([np_loser_board, np_loser_move_matrix], axis=1)
    return np_dataset


def create_loser_move_matrix(df_move_matrix, df_loser_index):
    loser_move_matrix = [
        df_move_matrix.values[i][j] for i, loser_index in enumerate(df_loser_index.values) for j in loser_index
    ]
    np_loser_move_matrix = np.array(loser_move_matrix)
    np_loser_move_matrix = np_loser_move_matrix.reshape(-1, 1, 8, 8)
    return np_loser_move_matrix


def create_loser_board(df_board, df_loser_index):
    loser_board = [
        df_board.values[i][j] for i, loser_index in enumerate(df_loser_index.values) for j in loser_index
    ]
    np_loser_board = np.array(loser_board)
    return np_loser_board


def split_dataset(np_dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must add up to 1"
    train_val, test = train_test_split(np_dataset, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)
    return train, val, test


def save_datasets(train, val, test):
    np.save("./data/train_loser_othello_dataset.npy", train)
    np.save("./data/val_loser_othello_dataset.npy", val)
    np.save("./data/test_loser_othello_dataset.npy", test)


if __name__ == "__main__":
    np_dataset = prepare_othello_dataset()
    train, val, test = split_dataset(np_dataset)
    save_datasets(train, val, test)
