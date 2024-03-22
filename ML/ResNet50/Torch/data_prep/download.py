from huggingface_hub import hf_hub_download
for i in range(5):
        hf_hub_download(repo_id="imagenet-1k", 
                filename=f"data/train_images_{i}.tar.gz", 
                local_dir="/home/eidf095/eidf095/crae-ml/imagenet-1k", 
                cache_dir="/home/eidf095/eidf095/crae-ml/imagenet-1k/.huggingface_cache", 
                repo_type="dataset")

hf_hub_download(repo_id="imagenet-1k", 
                filename=f"data/val_images.tar.gz", 
                local_dir="/home/eidf095/eidf095/crae-ml/imagenet-1k", 
                cache_dir="/home/eidf095/eidf095/crae-ml/imagenet-1k/.huggingface_cache", 
                repo_type="dataset")
