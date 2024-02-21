import os
from argparse import ArgumentParser

import torchaudio


def list_files(directory,cnt,exceed):
    for root, dirs, files in os.walk(directory):
        print(f'当前目录: {root}')

        for file in files:
            file_path = os.path.join(root, file)

            waveform, sample_rate = torchaudio.load(file_path)
            num_samples = waveform.size(1)
            time_per_sample = 1.0 / sample_rate
            audio_duration = num_samples * time_per_sample
            if exceed != 0 and audio_duration > exceed:
                os.remove(file_path)
                print(f'remove {file_path}')
            else:
                fs = file.split(".")
                new_path = os.path.join(root,f'{cnt}.{fs[-1]}')
                os.rename(file_path,new_path)
                print(f'文件: {file_path} -> {new_path}')
                cnt+=1

        for subdir in dirs:
            list_files(os.path.join(root, subdir),cnt+1)

if __name__ == '__main__':
    parser = ArgumentParser(description='rename data')
    parser.add_argument('-d', '--dir',
                        help='dir path',
                        type=str, default='data')
    parser.add_argument('-e', '--exceed',
                        help='clear file exceeding the maximum duration.Set to 0 to disable it.',
                        type=int, default='0')
    args = parser.parse_args()
    list_files(args.dir,0,args.exceed)
