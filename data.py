
import os
import numpy as np 
import soundfile as sf
import librosa
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
from keras.utils import np_utils
from tqdm import tqdm
import random

def hop_size(shape, target_len):
    return int((shape[0]+1)/float(target_len))

def get_mfccs(x):
    hop = hop_size(x.shape, 28)
    v_mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=28, hop_length=hop)
    v_mfcc = v_mfcc[:, :28]
    return v_mfcc # [28,28]

class AudioDigitDataLoader:

    def __init__(self, dataset_dir, num_digits=10, num_clips=20):
        self.dataset_dir = dataset_dir
        self.all_speakers = self.get_all_speakers()
        self.num_digits = num_digits
        self.num_clips = num_clips
        self.data = None
    
    def data_path(self, speakerID, digit, clip):
        # will produce something like '_dataset/000/5-003.ogg'
        speakerID = int(speakerID)
        speakerID = str(speakerID).zfill(3) # add leading 0's like 007
        clip = str(clip).zfill(3) # same
        filename = f'{digit}-{clip}.ogg'
        path = os.path.join(self.dataset_dir, speakerID, filename)
        return path

    def load_one_speaker_data(self, speaker, preprocessor=get_mfccs, augmentation=None):
        # Load all the data into a single np.array (and labels in another)
        total_clips = self.num_digits * self.num_clips
        count = 0
        min_len = 0
        data, labels = [], [] 
        for digit in range(self.num_digits):
            for clip in range(self.num_clips):
                path = self.data_path(speaker, digit, clip)
                x, sr = sf.read(path)
                if augmentation is not None:
                    xlist = augmentation(x)
                    data += [preprocessor(x) for x in xlist]
                    labels += [digit] * len(xlist)
                else:
                    ff = preprocessor(x)
                    data.append(ff)
                    labels.append(digit)
                count += 1
        return data, labels

    def load_all_speakers_data(self, speakers=None, preprocessor=get_mfccs, augmentation=None):
        if speakers is None:
            speakers = self.all_speakers
        data = []
        labels = []
        # return array of tuples (data, labels) for every speaker
        num_speakers = len(speakers)
        for speaker_index in tqdm(range(num_speakers)):
            speaker = speakers[speaker_index]
            d, l = self.load_one_speaker_data(speaker, preprocessor, augmentation)
            data.append(d)
            labels.append(l)
        return data, labels

    def get_all_speakers(self):
        def isSpeaker(speaker):
            return speaker.isdigit()
        all_speakers = sorted(os.listdir(self.dataset_dir))
        all_speakers = list(filter(isSpeaker, all_speakers))
        return all_speakers

    def get_data_and_labels(self, all_data, all_labels, speakers):
        speaker_count = len(speakers)
        data_per_speaker = self.num_digits * self.num_clips
        example_count = speaker_count * data_per_speaker
        data   = np.zeros([example_count, all_data.shape[-2], all_data.shape[-1]])
        labels = np.zeros([example_count])
        for i,speaker in enumerate(speakers):
            speaker = int(speaker)
            data_start = i * data_per_speaker
            data_end = data_start + data_per_speaker
            data[data_start:data_end] = all_data[speaker]
            labels[data_start:data_end] = all_labels[speaker]
        return data, labels, 

    def collect_data(self, data, max_length):
        # input format [num](length, dims), organized into [num, max_length, dims]
        np_data = np.zeros([len(data), max_length, data[0].shape[1]])
        lengths = np.zeros([len(data)])
        for j in range(len(data)):
            ll = np.minimum(data[j].shape[0], max_length)
            np_data[j][:ll] = data[j][:ll]
            lengths[j] = ll
        return np_data, lengths

    def get_data_and_labels_and_lengths(self, all_data, all_labels, speakers, max_length):
        speaker_count = len(speakers)
        example_count = 0
        for i,speaker in enumerate(speakers):
            example_count += len(all_data[int(speaker)])
        data   = np.zeros([example_count, max_length, all_data[0][0].shape[1]])
        labels = np.zeros([example_count])
        lengths = np.zeros([example_count])
        for i,speaker in enumerate(speakers):
            speaker = int(speaker)
            data_per_speaker = len(all_data[speaker])
            data_start = i * data_per_speaker
            data_end = data_start + data_per_speaker
            data[data_start:data_end], ll = self.collect_data(all_data[speaker], max_length)
            labels[data_start:data_end] = all_labels[speaker]
            lengths[data_start:data_end] = ll
        return data, labels, lengths

    def get_split_data(self, data, labels, num_train_speakers=18, cut_to_max_length=True, max_length=10, seed=None):
        all_speakers = self.all_speakers.copy()
        if seed is not None:
            random.seed(seed)
            random.shuffle(all_speakers)
            print(all_speakers)
        training_speakers = all_speakers[:num_train_speakers]
        testing_speakers = all_speakers[num_train_speakers:]
        if cut_to_max_length:
            all_length = [[d.shape[0] for d in spk_data] for spk_data in data]
            max_length = np.max(all_length)
        train_data, train_labels, train_lengths = self.get_data_and_labels_and_lengths(data, labels, training_speakers, max_length)
        test_data,  test_labels, test_length  = self.get_data_and_labels_and_lengths(data, labels, testing_speakers, max_length)
        return [train_data, train_labels, train_lengths], [test_data,  test_labels, test_length]

class AudioDigitDataset(keras.utils.Sequence):
    def __init__(self, data, labels, lengths, batch_size=16):
        idxs = np.argsort(lengths)
        self.data = data[idxs]
        self.labels = labels[idxs]
        self.lengths = lengths[idxs].astype(np.int32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        max_len = np.max(self.lengths[indices])
        return self.data[indices, :max_len], self.labels[indices]
