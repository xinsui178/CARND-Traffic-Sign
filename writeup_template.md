# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/3.jpg "Visualization for sign 3"
[image1_bar]: ./examples/bar_plot.jpg "Visualization for class count"

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/14_stop.png "Traffic Sign 1"
[image5]: ./examples/16_No_entry_3point5t.png "Traffic Sign 2"
[image6]: ./examples/33_turn_right.png "Traffic Sign 3"
[image7]: ./examples/34_turn_left.png "Traffic Sign 4"
[image8]: ./examples/3_speed_for_60.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/xinsui178/CARND-Traffic-Sign/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
 
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a example plot for the class 3.Samples for other classes can be found in the example folder.
![alt text][image1]
It is a bar chart showing how the class distributes among the train dataset, valid dataset and test dataset. It can be seen they share high similarity in term of the ratio of class distribution.
![alt text][image1_bar]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As the only step for preprocessing, I normalized the image data because I would like to make the model of higher accuracy when it applies to other dataset which means higher geralization. What I did is just divide image array by 255.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 5x5x16   	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Fully connected		| Input = 400. Output = 120.        			|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84.        			|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43.        			|
| Softmax				|       									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer of AdamOptimizer, learning rate starts from 0.001, batch size of 16 and epochs of 10 to prevent overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994.
* validation set accuracy of 0.939.
* test set accuracy of 0.93.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first chose the leNet cause I encouter this network in the tutorial. I feel it should be sufficient for this kind of problem.

* What were some problems with the initial architecture?
The input of the intial architecture is suitable for the grayscale while I need to modifiy it to RGB chanel.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I didnt adjust much about the architecture but more about twisting the hyperparameter.

* Which parameters were tuned? How were they adjusted and why?

I changed the epochs and the batch size, and I found the having relatively lower batch size sometimes can help with the accuracy improvement.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

From instinctive prospective, i feel dropout more fit to deeper model such as with 50+layers.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Lenet is sufficient for the traffic sign classification with only two conv layers and the performance is ok with good generalization as well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the [web](https://routetogermany.com/drivingingermany/road-signs):
![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The quality of the data is very good from the official website. I searched the signs selecting from the training dataset classes to ensure it is potentially can be correctly classified by the model. The challenging is that the image is of large size and I need to downsize it to fit with the network. However, the model works very well with the new web data.

- The 1 image result is correct:
The model predict class is 14
The true label is 14
- The 2 image result is correct:
The model predict class is 16
The true label is 16
- The 3 image result is correct:
The model predict class is 33
The true label is 33
- The 4 image result is correct:
The model predict class is 34
The true label is 34
- The 5 image result is wrong: This one might be challenging since 5 and 6 looks very similar.
The model predict class is 2
The true label is 3
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No_entry_3.5t         | No_entry_3.5t         						|
| Turn right			|  Turn right									|
| Turn left	      		| Turn left	   				 			    	|
| Speed for 60			| Speed for 50	  						    	|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| 2.02145006e-12        | Speed limit (30km/h)							|
| 6.59170364e-13		| Traffic signals								|
| 6.17723320e-13	    | No vehicles				 				    |
| 1.45343332e-14	    | Priority road     							|


For the second image, the model is relatively sure that this is a Turn right ahead sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99580204e-01	    | Vehicles over 3.5 metric tons prohibited		| 
| 4.19853575e-04        | Roundabout mandatory						    |
| 1.06645595e-10		| End of no passing							    |
| 1.15337805e-17	    | General caution		 				        |
| 2.30394038e-23	    | No passing				                    |


For the third image, the model is relatively sure that this is a Turn right ahead sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0			        | Turn right ahead						     	| 
| 2.23639017e-15        | Keep left							            |
| 1.77075639e-16		| Ahead only						    		|
| 2.19872779e-17	    | Roundabout mandatory				 		    |
| 8.39622443e-20	    | Speed limit (70km/h)							|


For the forth image, the model is relatively sure that this is a Turn right ahead sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0	                | Turn left ahead	                        	| 
| 2.44833396e-13        | Ahead only			                	    |
| 4.23099098e-15		| End of no passing						        |
| 9.50986588e-17	    | Go straight or left		 				    |
| 3.05261972e-17	    | Roundabout mandatory				            |


For the fifth image, the model is relatively sure that this is a Turn right ahead sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0			        | Speed limit (50km/h)						    | 
| 5.53310740e-17        | Speed limit (30km/h)						    |
| 1.22320193e-25		| Wild animals crossing					    	|
| 9.92739171e-26	    | Speed limit (80km/h)			 		        |
| 1.01945769e-32	    | Speed limit (60km/h)							|


