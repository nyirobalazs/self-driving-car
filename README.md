# Modified NVIDIA-based end-to-end self-driving model

In this paper, my colleague and I implemented a modified NVIDIA behavioral cloning model using a SunFounder PiCar-V toy car. In order to enable the vehicle to navigate safely in a total of 12 different situations during the live test, the neural network was trained using pre-recorded data. Thanks to this, a relatively low validation loss of 0.0126 was achieved. Furthermore, the car was able to successfully perform most of the tasks, except for the red traffic light, the consideration of direction arrows and the execution of a turn. 
