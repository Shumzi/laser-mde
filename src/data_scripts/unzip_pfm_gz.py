import os
import shutil
import gzip
from tqdm import tqdm


def main():
    path = os.path.join('../../data', 'geoPose3K')
    folders = (os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)))
    for folder in tqdm(folders, unit='folder'):
        for file in os.listdir(folder):
            if file.endswith('.gz'):
                with gzip.open(os.path.join(folder, file), 'rb') as f_in:
                    with open(os.path.join(folder, file.strip('.gz')), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    main()
