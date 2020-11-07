# McGill_Hackaton_2020
Using OpenCV to make low pixelated images into high pixelated ones.

Using this blog as an inspiration and as guidance with our project https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066

We want to increase the resolution of pictures coming from space, using some of the latest AI algorithms.

We are using OpenCV to upscale the image.

We are feeding the image to different pre trained models.

Testing 4 different models “edsr”, “fsrcnn”, “lapsrn”, “espcn”


source: https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres

EDSR
Trained models can be downloaded from here.

Size of the model: ~38.5MB. This is a quantized version, so that it can be uploaded to GitHub. (Original was 150MB.)
This model was trained for 3 days with a batch size of 16
Link to implementation code: https://github.com/Saafke/EDSR_Tensorflow
x2, x3, x4 trained models available
Advantage: Highly accurate
Disadvantage: Slow and large filesize
Speed: < 3 sec for every scaling factor on 256x256 images on an Intel i7-9700K CPU.
Original paper: Enhanced Deep Residual Networks for Single Image Super-Resolution [1]
ESPCN
Trained models can be downloaded from here.

Size of the model: ~100kb
This model was trained for ~100 iterations with a batch size of 32
Link to implementation code: https://github.com/fannymonori/TF-ESPCN
x2, x3, x4 trained models available
Advantage: It is tiny and fast, and still performs well.
Disadvantage: Perform worse visually than newer, more robust models.
Speed: < 0.01 sec for every scaling factor on 256x256 images on an Intel i7-9700K CPU.
Original paper: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network [2]
FSRCNN
Trained models can be downloaded from here.

Size of the model: ~40KB (~9kb for FSRCNN-small)
This model was trained for ~30 iterations with a batch size of 1
Link to implementation code: https://github.com/Saafke/FSRCNN_Tensorflow
Advantage: Fast, small and accurate
Disadvantage: Not state-of-the-art accuracy
Speed: < 0.01 sec for every scaling factor on 256x256 images on an Intel i7-9700K CPU.
Notes: FSRCNN-small has fewer parameters, thus less accurate but faster.
Original paper: Accelerating the Super-Resolution Convolutional Neural Network [3]
LapSRN
Trained models can be downloaded from here.

Size of the model: between 1-5Mb
This model was trained for ~50 iterations with a batch size of 32
Link to implementation code: https://github.com/fannymonori/TF-LAPSRN
x2, x4, x8 trained models available
Advantage: The model can do multi-scale super-resolution with one forward pass. It can now support 2x, 4x, 8x, and [2x, 4x] and [2x, 4x, 8x] super-resolution.
Disadvantage: It is slower than ESPCN and FSRCNN, and the accuracy is worse than EDSR.
Speed: < 0.1 sec for every scaling factor on 256x256 images on an Intel i7-9700K CPU.
Original paper: Deep laplacian pyramid networks for fast and accurate super-resolution [4]
Benchmarks
Comparing different algorithms. Scale x4 on monarch.png (768x512 image).

Inference time in seconds (CPU)	PSNR	SSIM
ESPCN	0.01159	26.5471	0.88116
EDSR	3.26758	29.2404	0.92112
FSRCNN	0.01298	26.5646	0.88064
LapSRN	0.28257	26.7330	0.88622
Bicubic	0.00031	26.0635	0.87537
Nearest neighbor	0.00014	23.5628	0.81741
Lanczos	0.00101	25.9115	0.87057


