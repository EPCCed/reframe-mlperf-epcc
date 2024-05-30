# Configurations

A lot of the configurations are pretty self explanatory so I wont go over them each configuration is slightly different due to using different optimizers and data etc.

### global_batch_size
The Global Batch Size is the is the amount of data passed forward and and backward propagated per parameter update across all devices

The global_batch_size is used to calculate the local batch size with the equation "lbs = gbs//num_gpus", which is the throughput per device per iteration.

If the dataset has 512 inputs and the global batch size(gbs) is 8 that would be 64 iterations per epoch, no matter the number of gpus.
 
If you have 4 gpus each GPU would pass 2 inputs into the model per iteration and then all reduce the gradients, causing 8 inputs to be passed into the model per parameter update.

### gradient_accumulation_freq

The gradient accumulation frequency is used to tell the trainloop how often the parameters should update

If you add gradient accumulation the number of iterations increases by the frequency of the gradient accumulation, but the parameters don't update every iteration, the model will update after the number of inputs that have been passed through the model across all devices is equal to the global batch size. the local batch size with gradient accumulation is calculated with the equation "lbs = (gbs//num_gpus)//grad_acum_freq" where the local batch size is the throughput per device per iteration.

If the dataset has 512 inputs and the global batch size(gbs) is 8 that would be 64 iterations per epoch, no matter the number of gpus.

If you have 4 gpus each GPU would pass 2 inputs into the model per iteration and then all reduce the gradients, causing 8 inputs to be passed into the model per parameter update.
 
If you had a gradient accumulation frequency of 2 on top of that, there would be 128 iterations per epoch, where each gpu would pass 1 input into the model per iteration and there would be a parameter update every 2nd iteration causing parameters to update after 8 inputs have been passed into the model.

### local_batch_size

Some benchmarks allow you to directly set the local batch size, this sets the gradient_accumulation_freq=1 and means each device will pass the local_batch_size into through the model per iteration causing a global batch size of num_gpus*local_batch_size.

### train_subset & val_subset

This setting allows you to configure the size of train and validation samples by setting them to 0 you will use the full dataset.

This allows you to benchmark quickly on a few thousand inputs.

### Dataloader args

**drop last**, **shuffle** and **prefetch** are all parameters passed directly to the `torch.utils.data.Dataloader`  see the Dataloader [documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

### device

The device argument allows to set the device used during training its recommend to leave it as `cpu` and when you want to use a gpu pass the `--device cuda` to the train.py, this allows you to use the same config for cpu and gpu jobs.

### opt section

To set the optimizer used in the job just change the name (e.g. `name: Adam`) not all optimizers are supported by each benchmark due to mlperfs regulations. The rest of the args under opt are passed directly to the torch.optim you chose via the name

### lr_schedule section 

Works the exact same as the opt section but is for the lr-scheduler .
