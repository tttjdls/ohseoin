import numpy as np
import pandas as pd
from config import Config
from util import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

class Freesound(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):

        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.data_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        data = data[np.newaxis, :]

        if self.mode is "train":
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data

    def _random_selection(self, file_path):

        input_length = self.config.audio_length
        # Read and Resample the audio
        data = load_data(file_path)

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data


class Freesound_logmel(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'
        file_path = os.path.join(self.config.data_dir, filename)
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        if self.mode is "train":
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data

    def _random_selection(self, file_path):

        input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)
        # Read the logmel pkl
        logmel = load_data(file_path)

        # Random offset / Padding
        if logmel.shape[2] > input_frame_length:
            max_offset = logmel.shape[2] - input_frame_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_frame_length + offset)]
        else:
            if input_frame_length > logmel.shape[2]:
                max_offset = input_frame_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_frame_length - logmel.shape[2] - offset)), "constant")
        return data


class ToTensor(object):

    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data


if __name__ == "__main__":
    config = Config(sampling_rate=22050,
                    audio_duration=1.5,
                    data_dir="../input/logmel+delta_w80_s10_m64")
    DEBUG = True

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/sample_submission.csv')

    LABELS = config.labels

    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")
    test.set_index("fname")
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    if DEBUG:
        train = train[:2000]
        test = test[:2000]

    skf = StratifiedKFold(n_splits=config.n_folds)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
        print("TRAIN:", train_split, "VAL:", val_split)
        train_set = train.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        print(len(train_set), len(val_set))

        trainSet = Freesound_logmel(config=config, frame=train_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = Freesound_logmel(config=config, frame=val_set,
                             transform=transforms.Compose([ToTensor()]),
                             mode="train")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        for i, (input, target) in enumerate(train_loader):
            print(i)
            print(input)
            print(input.size())
            print(target)
            break