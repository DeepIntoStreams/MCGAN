
# Moving MNIST Experiment


Moving MNIST [782Mb] contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame, the ```.npy``` file can be downloaded from [here](http://www.cs.toronto.edu/~nitish/unsupervised_video/).

After you download the .npy data, put it under folder ./data then run 
```
python train.py -use_cuda
```
If you have no GPU, ignore the ```-use_cuda``` keyword.
