**Traffic Sign Recognition** 

##Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./code_img/distribution.png "Visualization"
[image2]: ./code_img/grey.png "Grayscaling"
[image3]: ./code_img/visualdata.png "Data Visualization"
[image4]: ./new_img/1.jpg "Traffic Sign 1"
[image5]: ./new_img/2.jpg "Traffic Sign 2"
[image6]: ./new_img/3.jpg "Traffic Sign 3"
[image7]: ./new_img/4.jpg "Traffic Sign 4"
[image8]: ./new_img/5.jpg "Traffic Sign 5"
[image9]: ./code_img/test_p1.png "Test Probability 1"
[image10]: ./code_img/test_p2.png "Test Probability 2"
[image11]: ./code_img/test_p3.png "Test Probability 3"
[image12]: ./code_img/test_p4.png "Test Probability 4"
[image13]: ./code_img/test_p5.png "Test Probability 5"
[image14]: ./code_img/conv_visual1.png "Conv layer 1 visualization"
[image15]: ./code_img/conv_visual2.png "Conv layer 2 visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/liuyangly25/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb).

There is also an html file which can be downloaded from this repo. Here is the link to the html [HTML code](https://github.com/liuyangly25/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.html).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution based on different signs.

![Sign Distribution][image1]

The numbers are not same for each sign. The smallest amount is 180, however, the largest is around 2000. This will be a factor resulting in different accuracy for each sign.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale image may performed better than 3-channel RGB image. It might has less noise than color image.

Samples of grayscale images shown below:

![Grayscale][image2]

However, I commented out my grayscale because it does not increase validation accuracy a lot. The result is same as the validation accuracy without grayscale.

So, my preprocessing step is to normalize image dataset. Normalization will help the optimizer minimize the cost and find the global minimum instead of the local minimum.

Here is some images from training set.

![Training Data][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten 				| Input 5x5x16, output 400						|
| Fully connected		| Input 400, output 120							|
| Dropout				| Dropout function 								|
| Fully connected		| Input 120, output 84							|
| Dropout				| Dropout function 								|
| Output 				| Input 84, output 43							|
| Softmax				| Final Activation Function						|

 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer. Learning rate is 0.0007, batch size is 64, number of epochs is 20, Dropout is 0.5 for full connection layers.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.961 
* test set accuracy of 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I used the LeNet architect at first without dropout. But the validation acurracy is not good. It was around 0.87. So, a dropout is added to the architecture with keep_prob = 0.5. Dropout actually prevent overfitting,it randomly drops the unit in my fully connection layers. This gives me a much better accurracy, that is ~0.95. The reason why keep_prob = 0.5 is that half units drops yield better validation accuracy. With keep_prob = 0.4, the accurary increased slowly when epoches interates, it need more epoches. With keep_prob = 0.6, the accurary increased faster when epoches interates, but finally accuracy is same as keep_prob = 0.5. So keep_prob = 0.5 is chosen for the final value.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
N/A

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![New Image 1][image4] ![New Image 2][image5] ![New Image 3][image6] 
![New Image 4][image7] ![New Image 5][image8]

The second image might be difficult to classify because if there is some noise in this speed limit sign. The number is hard to classify. Compared with stop sign, which is the unique sign, speed limits signs is a group of similar signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70km/h  				| 60km/h 										|
| Right-of-way			| Right-of-way									|
| Priority road 		| Priority road   								| 
| No vehicles	  		| No vehicles					 				|
| stop					| stop      									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 52nd cell of the Ipython notebook, and the chart is in the 53th cell.

The certainty can be seen from the chart below.

(1). Image 1 - 70km/h

![Test Probability 1][image9]

For the Image 1, the model is has a wrong prediction of 60km/h (probability of 0.806), but the image does contain a 70km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .806         			| 60km/h 	  									| 
| .178     				| Bicycles crossing 							|
| .005					| Wild animal crossing							|
| .004	      			| Road working					 				|
| .003				    | No Vehicles 	     							|

(2). Image 2 - Right-of-way

![Test Probability 2][image10]

For the Image 2, the model is pretty sure that this is a Right-of-way sign (probability of 0.999), and the image does contain a Right-of-way sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Right of way   								| 
| .000     				| Pedestrians 									|
| .000					| Double Curve									|
| .000	      			| Beware of ice/snow			 				|
| .000				    | Slippery Road      							|

(3). Image 3 - Priority road

![Test Probability 3][image11]

For the Image 3, the model is relatively sure that this is a Priority road sign (probability of 1), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Priority road   								| 
| .00     				| End of no passing 							|
| .00					| No entry										|
| .00	      			| Roundabout mandatory							|
| .00				    | End of all speed and passing limits      		|

(4). Image 4 - No vehicles

![Test Probability 4][image12]

For the Image 4, the model is pretty sure that this is a No vehicles (probability of 0.948), and the image does contain a No vehicles sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .948         			| No vehicles   								| 
| .027     				| 50km/h 										|
| .018					| 30km/h										|
| .002	      			| 70km/h						 				|
| .001				    | Turn right ahead     							|

(5). Image 5 - Stop

![Test Probability 5][image13]

For the Image 5, the model is relatively sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop sign   									| 
| .000     				| No entry 										|
| .000					| Road work										|
| .000	      			| End of all speeds and passing limits			|
| .000				    | Go straight of right      					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![Conv1 Visual][image14]

![Conv2 Visual][image15]
