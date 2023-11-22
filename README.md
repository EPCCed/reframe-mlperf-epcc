# chris-ml-intern
Documentation and code for ML benchmarking and reframe testing

Task Type | Task | Model Name | Dataset Name | No. Model Parameters | Memory required to train Model (MB) | Uncompressed Dataset Size (GB)
:---: | :---: | :---: | :---: | :---: | :---: | :---:
ML | Natural Language Processing | Bert-Large | Wikipedia 2020/01/01 | 335,174,458 | 5,363 | 365
ML | Large Language Model | GPT-J | C4 | 6,053,381,344 | 96,854 | 750
ML | Image Classification | ResNet50 v1.5 | ImageNet-1k | 25,557,032 | 409 | 400
ML HPC | Cosmology Parameter Prediction | CosmoFlow | CosmoFlow N-body simulation | 8,907,556 | 71 | 5,100
ML HPC | Climate segmentation | DeepLabV3+ | CAM5+TECA | 56,454,720 | 903 | 8,800
ML HPC | Protein Folding | AlphaFold2 | OpenProteinSet and Protein Data Bank | "92,400,000" | "1478.4" | 2600


Memory required to train model doesn't take into account intermediate calculations of activations that are stored in memory for use in the backward pass. 
Memory required also assumes no AMP training and the use of single precision floating points 

# Implementation Explanation

Each benchmark works the same. First all arguments are defined in the config.yaml, then the global context singleton class reads the config and is used to retrieve config information by each component of the training loop. each benchmarks train loop is defined in the 'train.py' and is run via a slurm job from the 'run.slurm' bash file.

The train.py always follows the same structure:

```python
# import standard and global libraries

...

from ML.gc import GlobalContext
gc = GlobalContext("/path/to/config.yaml")

# import other local libraries

...

torch.manual_seed(0)
    if dist.is_mpi_available():
        backend = "mpi"
    elif gc.device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)

    if gc.device == "cuda":
        local_rank = os.environ["LOCAL_RANK"]
        torch.cuda.set_device("cuda:" + local_rank)

    # define dataloaders
    train_dataloader, val_dataloader = ...

    # define model and parallelize
    model = ...
    if gc.world_size > 1:
        # parallelize model
        ...
    
    # define optimizer from given optimizer options in config.yaml
    optimizer = ...
    
    # define lr_scheduler from given lr_schedule options in config.yaml
    lr_scheduler = ...

    # define criterion if applicable (BERT's loss is generated within the model.forward())
    criterion = ...

    epoch = 0

    while True:
        model.train()
        for x, y in train_dataloader:
            opt.zero_grad()
            logits = model.forward(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for x, y in val_dataloader:
                logits = model.forward(x)
                # get score based on quality target
                ...
        
        epoch +=1
        lr_scheduler.step()
        if score > gc["training"]["target"] or epoch == gc["data"]["n_epochs"]:
            break
```

Each implementation differs slightly from the above baseline as there are different rules for each benchmark as well as different example implementations, but the structure remains true.

