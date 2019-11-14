from util import *
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from config import *
import librosa

def get_wavelist():
    train_dir = 'C:/Users/TSP/dcase5/input/audio_train'
    test_dir = 'C:/Users/TSP/dcase5/input/audio_test'
    waves_train = sorted(os.listdir(train_dir))
    waves_test = sorted(os.listdir(test_dir))
    print(len(waves_train)+len(waves_test))
    df_train = pd.DataFrame({'fname': waves_train})
    df_train['train0/test1'] = pd.DataFrame(0 for i in range(len(waves_train)))

    df_test = pd.DataFrame({'fname': waves_test})
    df_test['train0/test1'] = pd.DataFrame(1 for i in range(len(waves_test)))

    df = df_train.append(df_test)
    df.set_index('fname', inplace=True)
    df.to_csv('./wavelist.csv')

def wav_to_logmel(wavelist):
    df = pd.read_csv(wavelist)
    pool = Pool(10)
    pool.map(tsfm_logmel, df.iterrows())

def tsfm_logmel(row):

    config = Config(sampling_rate=22050, n_mels=64, frame_weigth=80, frame_shift=10)

    item = row[1]
    p_name = os.path.join('C:/Users/TSP/dcase5/logmel+delta_w80_s10_m64', os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        if item['train0/test1'] == 0:
            file_path = os.path.join('C:/Users/TSP/dcase5/input/audio_train/', item['fname'])
        elif item['train0/test1'] == 1:
            file_path = os.path.join('C:/Users/TSP/dcase5/input/audio_test/', item['fname'])

        data, sr = librosa.load(file_path, config.sampling_rate)

        # some audio file is empty, fill logmel with 0.
        if len(data) == 0:
            print("empty file:", file_path)
            logmel = np.zeros((config.n_mels, 150))
            feats = np.stack((logmel, logmel, logmel))
        else:
            print(config.n_fft)
            melspec = librosa.feature.melspectrogram(data, sr,
                                                     n_fft=config.n_fft, hop_length=config.hop_length,
                                                     n_mels=config.n_mels)

            logmel = librosa.core.power_to_db(melspec)

            delta = librosa.feature.delta(logmel)
            accelerate = librosa.feature.delta(logmel, order=2)

            feats = np.stack((logmel, delta, accelerate)) #(3, 64, xx)

        save_data(p_name, feats)

if __name__ == '__main__':
    make_dirs()
    config = Config(sampling_rate=22050, n_mels=64, frame_weigth=80, frame_shift=10)
    get_wavelist()

    wav_to_logmel('wavelist.csv')