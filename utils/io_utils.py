import os

def getFileList(directory):
    fileList = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            fileList.append(file_path)

        for subdir in dirs:
            fileList += getFileList(os.path.join(root, subdir))
    return fileList