# How To Downlaod Imagenet-1k


## Setup
the easiest and **fastest** way to download imagenet is from the huggingface hub.

In order to download the dataset you will need to login to your a huggingface account via the huggingface-cli which can be installed with pip. I recommend doing this within a miniconda env, see how to set up a miniconda enviroment in `HOW_TO/build_torch` 

Once pip is available you can run `pip install --upgrade huggingface_hub` and once installed `huggingface-cli login` where you can insert your access token which you can create at https://huggingface.co/settings/tokens (the token should be in read mode).

After you have connected your access token to the huggingface-cli its time to setup our directories in your desired /work/ location create a file dir for the data (e.g. /work/z043/imagenet-1k) and within that dir run `mkdir .huggingface_cache` if you don't do this the script will use the.huggingface_cache in your /home dir which doesn't have enough storage to run the script

## Scripts
Next navigate to `ML/ResNet50/Torch/data_prep`, edit `download.py` so that the `local_dir` and `cache_dir` point to the correct location. Then you can run `python download.py` on a login node to download the data (there was an implementation that parallelized the download but it caused problems with the downloaded file due to the way huggingface streams in the data).

You'll notice that the data is in `/work/.../imagenet-1k/data` that is normal. Within the data dir run `mkdir train` and `mkdir val`, now back in `ML/ResNet50/Torch/data_prep` you'll find `data_prep.py` and `val_data_prep.py` change the `base` variable the the correct dir (**make sure that /train and /val are at the end of the base file string assigned to the base var as shown in the example string in `data_prep.py` and `val_data_prep.py`**)

## Data Pre-Processing

Within `ML/ResNet50/Torch/data_prep` there is a data.slurm enter the correct slurm cod, change the DATADIR to point to the correct location (e.g. /work/z043/shared/imagenet-1k/data) and also update the miniconda lines to activate your miniconda env. Finally run `sbatch data.slurm`

## Experimental

Within `data_prep.py` there is a variable `n` which is the number of workers in theory you can set this to up to the number of available cpu cores to speed up the work load.