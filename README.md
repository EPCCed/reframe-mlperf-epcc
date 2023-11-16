# chris-ml-intern
Documentation and code for ML benchmarking and reframe testing

Task Type | Task | Model Name | Dataset Name | No. Model Parameters | Memory required to train Model (MB) | Uncompressed Dataset Size (GB)
:---: | :---: | :---: | :---: | :---: | :---: | :---:
ML | Natural Language Processing | Bert-Large | Wikipedia 2020/01/01 | 335,174,458 | 5,363 | 365
ML | Large Language Model | GPT-J | C4 | 6,053,381,344 | 96,854 | 750
ML | Image Classification | ResNet50 v1.5 | ImageNet-1k | 25,557,032 | 409 | 400
ML HPC | Cosmology Parameter Prediction | CosmoFlow | CosmoFlow N-body simulation | 8,907,556 | 71 | 5,100
ML HPC | Climate segmentation | DeepLabV3+ | CAM5+TECA | 56,454,720 | 903 | ?
ML HPC | Protein Folding | AlphaFold2 | OpenProteinSet and Protein Data Bank | "92,400,000" | "1478.4" | 2600


Memory required to train model doesn't take into account intermediate calculations of activations that are stored in memory for use in the backward pass. 
Memory required also assumes no AMP training and the use of single precision floating points 