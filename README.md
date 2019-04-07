# Extended Kalman Filter  

[Reference](http://home.wlu.edu/~levys/kalman_tutorial/)  

## This repo is mainly for demonstration of the algorithm of  

1. Linear Kalman Filter  
2. Extended Kalman Filter  

## Brief  

* Kalman filter is a linear filter of a sequence data to do a tradeoff of 

1. Prediction from previous state  
2. Measurement from observation  
By calculating the Kalman Gain.

To Use of Kalman Filter, you must have some knowledge of your data.
```
X_t - Current State.
A - Linear State Transition Matrix.
U_t - Control State.
B - Linear Control Transition Matrix.
C - Linear State to Sensor Transition Matrix.
Z_t - Current Observation from different sensors.
R - Sensor constant error 
Q - Processing noise for both sensor
```

* Extended Kalman Filter is a extension package of KF by putting Transition Matrix to Transition function. It inherits the Kalman gain and update procedures.

To Use of EKF, you must have some knowledge of your data.
```
X_t - Current State
f - State Transition function of X_t
F - Jacobian Matrix of State Transition function
h - Observation Transition function of X_t
H - Jacobian Matrix of Observation Transition function
Z_t - Current Observation from different sensors
R - Sensor constant error 
Q - Processing noise for both sensor
```

## Example  

Example1 to example3 use of Linear Kalman Filter for examples in the tutorials

Example4 to Example6 use of Extended Kalman Filter for them.

**Would update for more concrete example**  
[ ] Mouse tracker Example
[ ] GPS Sensor fusion Example
[ ] Other Examples.

