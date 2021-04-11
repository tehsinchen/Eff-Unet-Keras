# Eff-Unet-keras

It is important to automatically segment the specific area which contains a certain feature in an image for only acquiring the information of interest. It both considerably saves time and effort compared to manually segmentation. There are some conventional algorithms that can be used to achieve such tasks, e.g, Canny and Sobel for edge detection. However, none of them are able to create a mask on a speckle-rich or high similarity between background and object image, e.g., bright-field and interferometric microscopy.
In this work, masks were predicted by efficient-unet (Eff-Unet) which refers to this paper "Eff-UNet: A Novel Architecture for Semantic Segmentation in Unstructured Environment" from bright-field images. The ground truths are the fluorescence images that indicate the area of cell nucleus and are binarized, see below.

![image](https://github.com/tehsinchen/Eff-Unet-keras/blob/main/input_data/input_data.png)

The EfficientNetB2 from keras was used as encoder in this case, others were the same, and the validation mIOU was 0.94.  Only rotation, flip, transpose were included in data augmentation. The kernel initializer was borrowed from https://github.com/zhoudaxia233/EfficientUnet. See below the predicted results (left is test images and right is the predicted masks).

![image](https://github.com/tehsinchen/Eff-Unet-keras/blob/main/results/cell_mask.gif)
