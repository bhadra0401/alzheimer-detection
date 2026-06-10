import os

data_path = "OriginalDataset"

for folder in os.listdir(data_path):
    print(folder, "->", len(os.listdir(os.path.join(data_path, folder))))