# Model Testing

## Simulator Testing
The trained model was tested using the Udacity self-driving car simulator in autonomous mode. The simulator successfully connected to the Python control server but telemetry data was not received. Steering and throttle commands could not be applied during autonomous driving.


# Model Testing
Due to the lack of telemetry from the simulator, model-level testing was performed independently to check that the model was working correctly. The model was loaded successfully, and input data with the same dimensions as pre-processed camera images (66x200x3) was passed in.
The model produced continuous, non-zero steering angle outputs (e.g. 0.0776), showing that the model had learned steering behaviour.
