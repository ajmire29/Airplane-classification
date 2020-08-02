# Airplane-classification

# Maximum Validation accuracy : 31.325 %

# Understanding of the Problem Statement
We are given a training dataset which has 70 families of airplanes in it. There are 100 variants of the airplanes in total. So, for one family there can be one or more variants.We need to classify the airplanes in the validation dataset according to their families. In the train folder, there are 100 subfolders corresponding to 100 variants.
So, we there can be two approaches to solve this problem.
1) First predict the variant of an airplane and then map the variant to the family using the variant-family mapping created before.
2) Directly predict the family of the airplane.

# Approach 1
Firstly, we created a mapping between variant and family using images_variant_train.txt and images_family_train.txt.

Then we preprocessed the training dataset. There are two variations in the size of the image as we had to try both grayscale and color images in the model. (28,28,1) and (32,32,3)
Rescaling the image between [0,1]. Extracting the images in the same order as the id's given in the images_family_train.txt( assuming that the first id i.e.1025794 corresponds to the first image in the folder 001 in "train". So, this complete numpy array covering all the folders in "train" is the X_train of the model.

For y_train & y_val, we extracted the second column of "images_variants_train.txt" and "images_variants_val.txt" respectively and then one-hot encoded them to feed them to the model.

# Approach 2
Preprocessing part remains the same. Only difference is in the y_train & y_val part, this time we are using directly "images_family_train.txt" and "images_family_val.txt".

# Models used
The major problem faced here is that of overfitting. Although in following models, Dropouts, BatchNormalization, Regularization, Data Augmentation has been used.

1) Pre-trained models
Used ResNet50 and VGG 19 but the number of parameters way too much for the given constraint of keeping parameters less than 3M.

2) Convolutional Autoencoders
Used a encoder-decoder network to first reconstruct the input and then using encoder & and its weights to do the classification task. But the model was not able to generalize to the dataset very well and gave a validation accuracy of 25%. But if more time is spent on this model, this can prove to be the best approach as it has less parameters and also it can give high accuracies.

3) 4 layered Convolutional Neural Network
This is the last approach that has been used here. A 4 ayered CNN with BatchNormalization, MaxPooling, Dropout has been used here. As the model was overfitting in the previous approaches, this model is made simpler aiming to solve the problem of overfitting. But still, this model could give maximum validation accuracy of 31.25%. I tried different combinations such as removing the data augmentation part(removing augmentation increased the training accuracy but it had no effect on validation accuracy), removing batch normalization(as it is debatable whethere batchnormalization should be applied before applying activation or after applying activation), adding more dropouts(for regularization), adding kernel regularizer in the CNN layer( for weight regularization) but all these changes had little effect on the validation accuracy.
