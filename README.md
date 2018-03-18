# WaveApp

#### Dependencies:

https://github.com/pytorch/audio



#### References:

https://github.com/jiecaoyu/pytorch_imagenet
https://github.com/D-X-Y/ResNeXt-DenseNet
http://www.festvox.org/cmu_arctic/


###How to use DataLoader for Arctic (Arctic.py) will generalize later 
change following variables depending on which dataset downloading for now
TODO: will reformat to have these values passed in as param and generalize later
Ex. for cmu_us_aew_arctic datset

    raw_folder = 'arctic/cmu_us_aew_arctic/raw'
    processed_folder = 'arctic/cmu_us_aew_arctic/processed'
    url = 'http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2'
    dset_path = 'cmu_us_aew_arctic/wav'
    processed_file = 'arctic_aew.pt'

    






## TODOs
0. Get this to work on AWS or something (not enough memory on my machine lol)
1. Integrate a logger (either TensorboardX or Tensorflow's Tensorboard).
2. Set up dataset for CMU Arctic and generic datasets.
3. Use a better architecture (much later).