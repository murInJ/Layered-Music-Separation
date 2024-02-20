import os
from argparse import ArgumentParser


def list_files(directory,cnt):
    for root, dirs, files in os.walk(directory):
        print(f'当前目录: {root}')

        for file in files:
            file_path = os.path.join(root, file)
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
    args = parser.parse_args()
    list_files(args.dir,0)
