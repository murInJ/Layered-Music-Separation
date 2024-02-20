import random

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.io_utils import *


def _get_audio(file_path):
    waveform,sample_rate = torchaudio.load(file_path)
    waveform = torch.mean(waveform,dim=0,keepdim=True)
    return {'waveform':waveform,'sample_rate':sample_rate}


def _split_array(array, size):
    return [array[i:i + size] for i in range(0, len(array), size)]

def normalize_waveform(waveform):
    max_amplitude = torch.max(torch.abs(waveform))
    return waveform / max_amplitude


def merge_waveforms(audio_list):
    # Check if any audio is provided
    if len(audio_list) == 0:
        raise ValueError("At least one audio must be provided")

    # Normalize and resample all waveforms
    max_len = max(len(audio['waveform'][0]) for audio in audio_list)
    max_sr = max(audio['sample_rate'] for audio in audio_list)
    merged_waveform = torch.zeros(1,max_len)
    for audio in audio_list:
        waveform = audio['waveform']
        samplerate = audio['sample_rate']
        # Resample if necessary
        if samplerate != max_sr:
            waveform = torchaudio.transforms.Resample(samplerate, max_sr)(waveform)
        # Normalize
        waveform = normalize_waveform(waveform)
        # Make the waveforms have the same length by zero-padding the shorter one
        len_diff = max_len - len(waveform[0])
        waveform = torch.nn.functional.pad(waveform, (0, len_diff))
        # Merge the waveforms
        merged_waveform += waveform

    return {'waveform': merged_waveform, 'sample_rate': max_sr}

class MusicDataset(Dataset):
    def __init__(self, root='./data',basicSize=8):
        self.file = getFileList(root)
        self.audios = [_get_audio(filePath) for filePath in self.file]
        self.basicSize = basicSize
    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        return merge_waveforms(self.data[idx]),self.data[idx]

    def generate_data(self):
        self.data = []
        random.shuffle(self.audios)
        arr = self.audios
        while len(arr) != 1:
            arr = _split_array(arr,self.basicSize)
            self.data += arr
            arr = [merge_waveforms(sub_arr) for sub_arr in arr]


class MixMusicDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )

    def __iter__(self):
        self.dataset.generate_data()
        for batch in super().__iter__():
            yield batch