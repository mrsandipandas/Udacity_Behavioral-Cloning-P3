# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture.png "Model Visualization"
[image2]: ./images/image_center.jpg "Center Image"
[image3]: ./images/image_left.png "Image from left camera"
[image4]: ./images/image_left_flipped.png "Flipped image from left camera"
[image5]: ./images/train_val_loss.png "Training and validation loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. My architecture is based on Nvidia [Pilotnet](https://arxiv.org/pdf/1704.07911.pdf). Here is a sneak peek into the Nvidia architecture.

![CNN architecture used for training the model][image1]

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 128 (model.py lines 100-120). At the end the model contains a few connected layers as inspired by the Nvidia Pilotnet architecture.

The model includes RELU layers to introduce nonlinearity (code line 107-111), and the data is normalized in the model using a Keras lambda layer (code line 102-104). 

#### 2. Attempts to reduce overfitting in the model

I had introduced dropout layers in order to reduce overfitting. However, I realised this introduces too much jittery in the steering angles and as a result in some situations I have observed that the car is unable to steer in tight turns. Hence, I have removed the dropout as of now, which is not ideal though. I plan to work on this issue more, given I find sufficeint time later on. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 121).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also driving in reverse direction in the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to calculate steering angles based on the three camera data. I calculated the offset correction for the right and left cameras manually (model.py line 56-59)

My first step was to use a convolution neural network model similar to the Nvidia Pilotnet. This model has been tested on the cars used by Nvidia with three mounted cameras on their GPU platform to compute the steering angles, speed and throttle. For our case, I was just interested inthe steering angles for running the vehicles in autonomous mode in the test track.

I also collected the data for both the training tracks and augmented the data using flipping the images.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle was running perfectly in track 1. However, in track 2 the vehicle sometimes fell off the track. May be adding some more training data with different brightness would help in running in the track with shadows in a more accurate manner without falling.


#### 2 Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid going off the road and come to the center if it drifts away. Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generate more training data. For example, here is an image that has then been flipped:

![Image from left camera][image3]
![Flipped image from left camera][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by veryfying the loss and validation loss. 
![Training Loss and Validation Loss in different epochs while training][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Finally, a video of how the car ran in the track in autonomous mode. This was prepared using collecting all the images used in the 'run1' folder.
```sh
python video.py run1
```

[![Video autonomous mode - First track](https://img.youtube.com/vi/BfCeOCKSXAg/0.jpg)](https://www.youtube.com/watch?v=BfCeOCKSXAg)

