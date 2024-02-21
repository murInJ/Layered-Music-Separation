import random
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.io_utils import *

def _split_array(array, size):
    return [array[i:i + size] for i in range(0, len(array), size)]

def _normalize_waveform(waveform):
    max_amplitude = torch.max(torch.abs(waveform))
    return waveform / max_amplitude

def _pad_waveform(waveform,targetLen):
    len_diff = targetLen - len(waveform[0])
    if len_diff <= 0:
        return waveform
    waveform = _normalize_waveform(waveform)
    pad_before = random.randint(0, len_diff)
    pad_after = len_diff - pad_before
    waveform = torch.nn.functional.pad(waveform, (pad_before, pad_after))
    return waveform
def _adjust_waveforms(unadjust_waveforms,targetLen=None):
    if len(unadjust_waveforms) == 0 :
        raise ValueError("At least one audio must be provided")

    pad_waveforms = []

    if targetLen is None:
        targetLen = max(len(wf[0]) for wf in unadjust_waveforms)
    for waveform in unadjust_waveforms:
        pad_waveforms.append(_pad_waveform(waveform,targetLen))

    return pad_waveforms
def _merge_waveforms(waveforms,sample_rates):
    return sum(_adjust_waveforms(waveforms)),sample_rates[0]



class MusicDataset(Dataset):
    def __init__(self, root='./data',basicSize=8,maxDataNum=1000,sample_rate=48000):
        self.file = getFileList(root)
        self.basicSize = basicSize
        self.maxDataNum = maxDataNum
        self.sample_rate = sample_rate
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (data_w,sample_rate),target_w = self._load_path_data(self.data[idx])
        return data_w,target_w,sample_rate

    def generate_data(self):
        self.data = []
        random.shuffle(self.file)
        arr = self.file
        while len(arr) != 1 and len(self.data) < self.maxDataNum:
            arr = _split_array(arr,self.basicSize)
            self.data += arr
        self.data = self.data[:self.maxDataNum]
    def _load_path_data(self,path_data):
        if isinstance(path_data, str):
            waveform, sample_rate = torchaudio.load(path_data)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
            return (waveform, sample_rate),None
        if isinstance(path_data,list):
            waveforms = []
            sample_rates = []
            for path in path_data:
                (waveform, sample_rate),_ = self._load_path_data(path)
                waveforms.append(waveform)
                sample_rates.append(sample_rate)
            return _merge_waveforms(waveforms,sample_rates),waveforms
class MixMusicDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        def collate_fn(batch):
            sample_rate = batch[0][2]
            batch_data_w = [item[0] for item in batch]
            batch_target_w = [item[1] for item in batch]

            maxLen = max(*[t.size(1) for t in batch_data_w],*sum([[t.size(1) for t in target] for target in batch_target_w],[]))
            data_ws = torch.stack(_adjust_waveforms(batch_data_w,maxLen))
            target_ws = [torch.stack(_adjust_waveforms(target_w,maxLen)) for target_w in batch_target_w]


            return data_ws,target_ws,sample_rate
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
            worker_init_fn=worker_init_fn,
        )

    def __iter__(self):
        self.dataset.generate_data()
        for batch in super().__iter__():
            yield batch