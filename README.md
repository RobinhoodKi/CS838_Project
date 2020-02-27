# CS 838/BMI 826 Final Project
Topic: Unsupervised Segmentation by super-pixel refinement and clustering
## Quick start
To evaluate our model performance, you can direcly execute:
```
python demo.py
```
|example 1 | output | example 2 | output |
|----------|--------|-----------|--------|
|<img src="https://github.com/RobinhoodKi/CS838_Project/blob/master/samples/9_3814.jpg" width="200">|<img src="https://github.com/RobinhoodKi/CS838_Project/blob/master/results/9_3814.bmp" width="200">|<img src="https://github.com/RobinhoodKi/CS838_Project/blob/master/samples/9_2965.jpg" width="200">|<img src="https://github.com/RobinhoodKi/CS838_Project/blob/master/results/9_2965.bmp" width="200">|

Detailed arguments:
```
python demo.py
[--weights_test]  # the path of model checkpoint      default: (str) model/checkpoint.pth.tar
[--test_dir]      # the directory of testing samples  default: (str) samples/*.jpg
[--output_dir]    # the directory to store output     default: (str) results/
[--is_crf]        # whether to use CRF or not         default: (int) 0
[--img_channe]    # number of input image channels    default: (int) 3
[--num_channel]   # number of network output channels    default: (int) 100
```
If there is any dependency issue, please find `requirement.txt` to check the environment or install the required libraries by:
```
pip install -r requirement.txt
```


## Training
Please find the `train.ipynb` to see our implementation and training details.

## Evaluation
Please find the `eval_IoU.ipynb` to see the quantitative results. Note that
we randomly sample three results from each method and then calculate their mIoU with the ground-truth labels.
