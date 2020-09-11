

# GSL

This is our PyTorch implementation for GSL. It is still under active development.

The code was written by [Yunhao Ge](yunhaoge@usc.edu) [Website](gyhandy.github.io) and [Gan Xin](gxin@usc.edu)
## Citation
If you use this code for your research, please cite:



## Code description

### datasets
You can create fonts and dsprites dataset with create_{}_dataset.py 
e.g.
- Create Fonts dataset，`create_fonts_dataset.py`

### How to start
A group of scripts(main, solver, model, dataset) form a complete task:
e.g. If you use Fonts dataset to achieve GSL, you will use 
`main_Nswap_fonts.py`
`solver_Nswap_fonts.py`
`model_share.py`
`dataset_Nswap_fonts.py`

