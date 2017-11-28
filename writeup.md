# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[german_30]: ./images/german_30.jpg "Speed limit (30km/h)"
[german_60]: ./images/german_60.jpg "Speed limit (60km/h)"
[german_cycle]: ./images/german_cycle.jpg "Bicycles crossing"
[german_road]: ./images/german_road.jpg "Road work"
[german_stop]: ./images/german_stop.jpg "Stop"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

[Link](https://github.com/anilmuthineni/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) to writeup. 

[Link](https://github.com/anilmuthineni/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) to jupyter notebook with all the code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

```python
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory analysis of the data set.

```python
Speed limit (20km/h): 180
Speed limit (30km/h): 1980
Speed limit (50km/h): 2010
Speed limit (60km/h): 1260
Speed limit (70km/h): 1770
Speed limit (80km/h): 1650
End of speed limit (80km/h): 360
Speed limit (100km/h): 1290
Speed limit (120km/h): 1260
No passing: 1320
No passing for vehicles over 3.5 metric tons: 1800
Right-of-way at the next intersection: 1170
Priority road: 1890
Yield: 1920
Stop: 690
No vehicles: 540
Vehicles over 3.5 metric tons prohibited: 360
No entry: 990
General caution: 1080
Dangerous curve to the left: 180
Dangerous curve to the right: 300
Double curve: 270
Bumpy road: 330
Slippery road: 450
Road narrows on the right: 240
Road work: 1350
Traffic signals: 540
Pedestrians: 210
Children crossing: 480
Bicycles crossing: 240
Beware of ice/snow: 390
Wild animals crossing: 690
End of all speed and passing limits: 210
Turn right ahead: 599
Turn left ahead: 360
Ahead only: 1080
Go straight or right: 330
Go straight or left: 180
Keep right: 1860
Keep left: 270
Roundabout mandatory: 300
End of no passing: 210
End of no passing by vehicles over 3.5 metric tons: 210


Label with maximum training examples: Speed limit (50km/h)(2010)
Label with minimum training examples: Speed limit (20km/h)(180)
```

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the noise in rgb color space due to lighting and time of the day.
As a second step, I normalized the grayscale data to be with mean 0 and standard deviation 1.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x128       									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Flatten  	      	    | outputs 3200 				|
| Fully connected		| outputs 500      									|
| Fully connected		| outputs 100    									|
| Fully connected		| outputs num_classes					|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size 100, epochs 10 and learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
```python
Train Accuracy = 1.000
Validation Accuracy = 0.965
Test Accuracy = 0.950
```

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I tried a basic lenet model with at most 32 filters. Later I increased the model size by increasing the filters in the convolution layers and the size of the fully connected layers.
* What were some problems with the initial architecture?

The initial architecture is very simple with high bias, to fit the complex problem.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Earlier model performed with low accuracy on both training and validation sets. It is an indication that the model has high bias. So, I increased the model size. To keep it close to original LeNet model, I haven't added any extra layers, but I increased the number of filters in convolution layers and the size of fully connected layers. 

If a well known architecture was chosen:
* What architecture was chosen?

The architecture is based on LeNet architecture originally designed for recognizing hand written digits.  
* Why did you believe it would be relevant to the traffic sign application?

Both handwritten digit recognition and traffic sign detection are similar problems, with similar input images and finite number of classes (10-40).
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Final model has a test accuracy of 0.95. This is a reasonably good model, though there is scope for improvement.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose 5 different traffic sign images of size 32x32

![Speed limit (30km/h)][german_30]

![Speed limit (60km/h)][german_60]

![Bicycles crossing][german_cycle]

![Road work][german_road]

![Stop][german_stop]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|Image|Prediction|
|:--:|:--:|
|Speed limit (30km/h) |Speed limit (60km/h) |
|Speed limit (60km/h) |Speed limit (60km/h) |
|Bicycles crossing |Bicycles crossing |
|Road work |Road work |
|Stop |Stop |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a "Speed limit (60km/h)", but the image contains a "Speed limit (30km/h)". The top five soft max probabilities were

|Probability|Prediction|
|:---:|:---:|
|1.000|Speed limit (60km/h)|)
|0.000|Speed limit (30km/h)|)
|0.000|Speed limit (80km/h)|)
|0.000|Speed limit (20km/h)|)
|0.000|Speed limit (50km/h)|)



For the second image, the model is very sure that this is a "Speed limit (60km/h)", and the image does contain a "Speed limit (60km/h)". The top five soft max probabilities were

|Probability|Prediction|
|:---:|:---:|
|1.000|Speed limit (60km/h)|)
|0.000|Speed limit (80km/h)|)
|0.000|Speed limit (20km/h)|)
|0.000|Speed limit (30km/h)|)
|0.000|Speed limit (50km/h)|)


For the third image, the model is very sure that this is a "Bicycles crossing", and the image does contain a "Bicycles crossing". The top five soft max probabilities were
|Probability|Prediction|
|:---:|:---:|
|1.000|Bicycles crossing|)
|0.000|Speed limit (20km/h)|)
|0.000|Speed limit (30km/h)|)
|0.000|Speed limit (50km/h)|)
|0.000|Speed limit (60km/h)|)


For the fourth image, the model is very sure that this is a "Road work", and the image does contain a "Road work". The top five soft max probabilities were
|Probability|Prediction|
|:---:|:---:|
|1.000|Road work|)
|0.000|Speed limit (20km/h)|)
|0.000|Speed limit (30km/h)|)
|0.000|Speed limit (50km/h)|)
|0.000|Speed limit (60km/h)|)


For the fifth image, the model is very sure that this is a "Stop", and the image does contain a "Stop". The top five soft max probabilities were
|Probability|Prediction|
|:---:|:---:|
|1.000|Stop|)
|0.000|Turn right ahead|)
|0.000|Speed limit (20km/h)|)
|0.000|Speed limit (30km/h)|)
|0.000|Speed limit (50km/h)|)

