# thesne
This repository is essentially the same as [this repository](https://github.com/paulorauber/thesne).  
The main contents of this repository are the following below:

1. This code has been rewritten using pytorch instead of theano.
2. This code has been rewritten so that you can read almost all variables.

The `thesne_using_theano` has exactly the same contents as original `thesne` just by rewriting the variables.  
The plotted figures in `result/gaussian_figure` are the experimental results obtained by performing the same experiment as the original paper using pytorch, and
you can see that the code has been rewritten normally.


## thesne_using_pytorch
### How to use
When you run `examples/gaussians.py`, you hove to change some variables, `dir_path` and `save_path`, to suit your environment.

### Using GPU
You can select your gpu device as `os.environ["CUDA_VISIBLE_DEVICES"]`.  
This code automatically run on cpu without you need to rewrite if you can't use gpu, but it is recommended to use gpu in terms of calculation time 