from __future__ import print_function
from data.data import DataProperty

#import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import torchaudio
import numpy as np


class Arctic(DataProperty):
    """ Arctic aew set 'http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2'
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions.
    """


    #can take these in as param for general dataloader




    def __init__(self, root, transform=None, target_transform=None, download=False, dev_mode=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.num_samples = 0
        self.max_len = 0
        self.datasets = ['bdl', 'slt', 'jmk', 'awb']
        self.processed_folder = 'arctic/processed'
        self.processed_file = 'arctic.pt'
        self.paths = []

        if download:
            for d in self.datasets:
                raw_folder = 'arctic/cmu_us_%s_arctic' % d
                url = 'http://festvox.org/cmu_arctic/packed/cmu_us_%s_arctic.tar.bz2' % d
                dset_path = 'cmu_us_%s_arctic/wav' % d
                dset_abs_path = os.path.join(
                    self.root, raw_folder, dset_path)
                self.paths.append(dset_abs_path)
                self.download(raw_folder,url,dset_abs_path)

            self.process()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        #import ipdb; ipdb.set_trace()
        self.data, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, self.processed_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        audio, target = self.data[index], self.labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_file))

    def download(self, raw_folder, url, dset_abs_path):
        """Download the arctic data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        raw_abs_dir = os.path.join(self.root, raw_folder)

        # download files
        try:
            os.makedirs(os.path.join(self.root, raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(self.root, raw_folder, filename)
        if not os.path.isfile(file_path):
            data = urllib.request.urlopen(url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
        else:
            print("Tar file already downloaded")
        if not os.path.exists(dset_abs_path):
            with tarfile.open(file_path) as zip_f:
                zip_f.extractall(raw_abs_dir)
        else:
            print("Tar file already extracted")
        # if not self.dev_mode:
        #     os.unlink(file_path)

    def process(self):
        # process and save as torch files
        print('Processing...')
        file = os.path.join(self.root, self.processed_folder, self.processed_file)
        if os.path.isfile(file):
            return
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        tensors = []
        labels = []
        lengths = []
        for j, p in enumerate(self.paths):
            audios = [x for x in os.listdir(p) if ".wav" in x]
            print("Found {} audio files".format(len(audios)))
            for i, f in enumerate(audios):
                full_path = os.path.join(p, f)
                sig, sr = torchaudio.load(full_path)
                tensors.append(sig)
                lengths.append(sig.size(0))
                #label = np.zeros(len(self.paths))
                #label[j] = 1
                label = j
                labels.append(label)
            # sort sigs/labels: longest -> shortest
        tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
            zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
        self.max_len = tensors[0].size(0)
        
        torch.save(
            (tensors, labels),
            os.path.join(
                self.root,
                self.processed_folder,
                self.processed_file
            )
        )

        print('Done!')
