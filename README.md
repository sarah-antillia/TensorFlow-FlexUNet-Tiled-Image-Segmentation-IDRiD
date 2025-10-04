<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-IDRiD (2025/10/04)</h2>

This is the first experiment of Image Segmentation for Indian Diabetic Retinopathy (IDRiD) Images, 
 based on our 
 <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
<b>TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass)</b></a>
, and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1bDoCilozlhvu0cTd14BVCLO5e46FEZfk/view?usp=sharing">
<b>Tiled-IDRiD-PNG-ImageMask-Dataset.zip</b></a> with colorized masks.
which was derived by us from 
<br><br>
<a href="https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid">
<b>
Indian Diabetic Retinopathy Image Dataset (IDRiD)
</b>
</a>
<!--
<b>Experiment Strategies: Divide And Quecker</b><br>
In this experiment, we employed the following strategies.
<br>
-->
<br>
<br>
<b>Divide-and-Conquer Strategy</b><br>
In this experiment with the TensorFlowFlexUNet segmentation model, 
since the images and masks in the training dataset are very large (4288x2848 pixels),
we adopted the following <b>Divide-and-Conquer Strategy</b> for building the segmentation model.
<br>
<br>
<b>1.Tiled Image and Colorized Mask Dataset</b><br>
We generated a master PNG Image and colorized mask datasets of 4288x2848 pixels from the training JPG files and all ground truth TIF files 
using the mask color-map
<b>(Microaneurysms:white,  Haemorrhages:red, Hard Exudates:blue, Soft Exudates:green, Optic Disc:yellow)</b>
, and then generated a 512x512 pixels Tiled IDRiD dataset from the master datasets by a tiledly splitting method. 
<!--
<table>
<tr>
<td>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/sample_image.png" width="512" height="auto">
</td
<td>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/sample_mask.png" width="512" height="auto">
</td
</table>
<br>
-->
<br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model by using the Tiled-IDRiD-ImageMask-Dataset, 
which was tiledly-splitted to 512x512 pixels 
and reduced to 512x512 pixels image and mask dataset from the master 4288x2848 pixels images and mask files.<br>
<br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict the colorized 
Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates, Optic Dis )c
legions for the mini_test images with a resolution of 4288x2848 pixels.<br><br>

<hr>
<b>Acutual Tiled Image Segmentation for IDRiD Images of 4288x2848 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but lack precision in some areas,.
<br>
<b>color_map (Microaneurysms:white,  Haemorrhages:red, Hard Exudates:blue, Soft Exudates:green, Optic Disc:yellow)</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_01.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_01.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_01.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_02.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_02.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_02.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_09.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_09.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_09.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>1 Dataset Citation</h3>
The dataset used here was obtained from the following <b>IEEE DataPort</b> web site<br>
<a href="https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid">
<b>
Indian Diabetic Retinopathy Image Dataset (IDRiD)
</b>
</a>
<br><br>
Please see also <a href="https://idrid.grand-challenge.org/">
<b>DIABETIC RETINOPATHY: SEGMENNTATION AND GRAND CHALLENGE</b> </a>
<br>
<br>
<b>Citation Author(s):</b><br>
Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, <br>
Vivek Sahasrabuddhe, Fabrice Meriaudeau,<br>
April 24, 2018, "Indian Diabetic Retinopathy Image Dataset (IDRiD)", IEEE Dataport, <br>
<br>
DOI: <a href="https://dx.doi.org/10.21227/H25W98">https://dx.doi.org/10.21227/H25W98</a><br>
<br>
<b>License:</b><br>
<a href="http://creativecommons.org/licenses/by/4.0/">
Creative Commons Attribution 4.0 International License.
</a>
<br>
<br>
<h3>
2 Tiled-IDRiD ImageMask Dataset
</h3>
<h4>2,1 Download Tiled IDRiD Dataset</h4>
 If you would like to train this Tiled-IDRiD Segmentation model by yourself,
 please download the 512x512 pixels dataset
<a href="https://drive.google.com/file/d/1bDoCilozlhvu0cTd14BVCLO5e46FEZfk/view?usp=sharing">
Tiled-IDRiD-PNG-ImageMask-Dataset.zip</a> on the google drive, 
expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-IDRiD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Tiled-IDRiD Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Tiled-IDRiD/Tiled-IDRiD_Statistics.png" width="512" height="auto"><br>
<br>

<!--
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
-->
<hr> 
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br>

<br>
<h4>2.2 Derivation dataset</h4>
The folder structure of the original Segmentation data is the following.<br>
<pre>
./A. Segmentation
├─1. Original Images
│  ├─a. Training Set
└─2. All Segmentation Groundtruths
   ├─a. Training Set
   │  ├─1. Microaneurysms
   │  ├─2. Haemorrhages
   │  ├─3. Hard Exudates
   │  ├─4. Soft Exudates
   │  └─5. Optic Disc
   └─b. Testing Set
       ├─1. Microaneurysms
       ├─2. Haemorrhages
       ├─3. Hard Exudates
       ├─4. Soft Exudates
       └─5. Optic Disc
</pre>
For simplicity, howerver, we generated a master dataset from the images and all groundtruths of <b> a. Training Set</b>.<br>
To generate a colorized mask corresponding to each image file, 
we merged the all categorized mask files by overlapping them to become a one mask file corresponding to each image file,
used the following colors map:<br>
<b>(Microaneurysms:white, Haemorrhages:red, Hard Exudates:blue, Soft Exudates:green, Optic Disc:yellow)</b>
<br><br>
Finally, we geneated our 512x512 pixels Tiled ImageMask Dataset from the master images and colorized mask files.<br> 

On the derivation of this tiled dataset, please refer to our repository:<br>
<a href="https://github.com/sarah-antillia/Tensorlfow-Tiled-Image-Segmentation-IDRiD-Haemorrhages">
<b>Tensorlfow-Tiled-Image-Segmentation-IDRiD-Haemorrhages</b></a>
<br>

<h4>2.3  Train Dataset Sample</h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Tiled-IDRiD TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Tiled-IDRiD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Tiled-IDRiD and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 6

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]

</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Tiled-IDRiD 1+5 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+5 classes
; RGB color   Microaneurysms:white,  Haemorrhages:red, Hard Exudates:blue, Soft Exudates:green, Optic Disc:yellow
rgb_map = {(0,0,0):0,(255,255,255):1,(255, 0, 0):2,    (0,0,255):3,         (0,255,0):4,        (255, 255, 0):5  }
</pre>

<b>Tiled Inference</b><br>
<pre>
[tiledinfer] 
overlapping = 64
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output_tiled/"
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTiledInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"

num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>images_dir</b> folder specified by <b>tiledinfer</b> section. 
 This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 41,42,43,44)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 84,85,86,87)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 87.<br><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/train_console_output_at_epoch87.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Tiled-IDRiD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Tiled-IDRiD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Tiled-IDRiD</b> folder,
 and run the following bat file to evaluate TensorFlowFlexUNet model for Tiled-IDRiD.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/evaluate_console_output_at_epoch87.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Tiled-IDRiD/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Tiled-IDRiD/test was not low, and dice_coef_multiclass 
not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1946
dice_coef_multiclass,0.934
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Tiled-IDRiD</b> folder
, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Tiled-IDRiD.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of IDRiD Images of 4288x2848 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_02.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_02.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_02.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_06.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_06.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_06.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_07.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_07.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_07.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_01.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_01.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_01.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_10.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_17.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/images/IDRiD_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test/masks/IDRiD_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Tiled-IDRiD/mini_test_output_tiled/IDRiD_20.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge</b><br>
Prasanna Porwal
, 
Samiksha Pachade, Manesh Kokare, Girish Deshmukh, Jaemin Son, Woong Bae, Lihong Liu<br>
, Jianzong Wang, Xinhui Liu, Liangxin Gao, TianBo Wu, Jing Xiao, Fengyan Wang<br>, 
Baocai Yin, Yunzhi Wang, Gopichandh Danala, Linsheng He, Yoon Ho Choi, Yeong Chan Lee<br>
, Sang-Hyuk Jung,Fabrice Mériaudeau<br>
<br>
DOI:<a href="https://doi.org/10.1016/j.media.2019.101561">https://doi.org/10.1016/j.media.2019.101561</a>
<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841519301033">
https://www.sciencedirect.com/science/article/abs/pii/S1361841519301033</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Tiled-Image-Segmentation-IDRiD-HardExudates</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-IDRiD-HardExudates">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-IDRiD-HardExudates
</a>
<br>
<br>

