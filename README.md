# Modified NVIDIA-based end-to-end self-driving model

In this paper, my colleague and I implemented a modified NVIDIA behavioral cloning model using a SunFounder PiCar-V toy car. In order to enable the vehicle to navigate safely in a total of 12 different situations during the live test, the neural network was trained using pre-recorded data. Thanks to this, a relatively low validation loss of 0.0126 was achieved. Furthermore, the car was able to successfully perform most of the tasks, except for the red traffic light, the consideration of direction arrows and the execution of a turn. 

The project description is available [HERE](https://github.com/nyirobalazs/self-driving-car/blob/main/assets/MLiS_Project_2022.pdf)
<br>
The paper from the project is available [HERE](https://github.com/nyirobalazs/self-driving-car/blob/main/Modified_NVIDIA_based_end_to_end_self_driving_model.pdf)

[![Self driving](./assets/self-drive-poster%20(1).jpg)]()

# Data 

- A total of 31,673 images were used to teach the network. 
- Out of these, 13,797 were taken from the public database provided for the challenge and another 17,896 images were collected by us.
- These are RGB images with a resolution of 320x240 pixels, and the file name of each image is used to record the steering position and speed at the time of capture. 
- For the data collection, the car was manually driven on the track so that 90% of the time it was as perfect in the lane as possible, but another 10% of the time data was collected where the car was driven from the edge of the track back to the correct position. This was necessary because if the self-driving function fails and the car finds itself off the track, the database would not normally cover the correction of this error. This way, there is a chance that the model can correct itself. 
- During data capture, the available elements (cubes, human figures, traffic lights and arrows) are alternated to best cover the positions and scenarios that can be rotated during testing. Furthermore, the data was split 80-20% between training and test datasets. 

# Model architecture

- The NVIDIA architecture is made up of five units, each with a convolution section and a max pooling layer, followed by a dropout. These blocks allow the network to extract features from the images. 
- This is finally followed by three fully connected layers that allow the extracted information to be mapped to steering angles and velocities. 
- In the final architecture, a batch normalization layer is added after each block, as a regularization method to normalize the data and thus allow more accurate prediction. 
- There were two separate output layers (one for velocity and one for steering angle), with sigmoid activation due to the regression property of the problem. - -Also as a consequence, the loss value was calculated using Mean Square Error (MSE).

# Result

![Video](https://github.com/nyirobalazs/self-driving-car/blob/main/assets/self_driving.gif)
