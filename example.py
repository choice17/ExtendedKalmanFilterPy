"""
Matplotlib keywait example
    #def quit_figure(event):
    #    if event.key == 't':
    #        plt.close(event.canvas.figure)
    #    elif event.key == 'p':
    #        print("hi I am here")

    #plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

"""

import numpy as np
from EKF import EKF

import argparse
import matplotlib.pyplot as plt
import time

def example1():
    # time t = 0 - 9
    # True state
    X_true = np.array([[1000, 750, 563, 422, 316, 237, 178, 133, 100, 75]])
    # Observation
    Z = np.array([[1090, 882, 554, 233, 345, 340, 340, 79, 299, -26]])

    ## Parameters ##
    # Sensor error
    R = np.array([[200]])
    # Prediction transition
    A = np.array([[0.75]])
    # Prediction Covariance init
    P = np.array([[1]])

    ## Initialize State ##
    ekf = EKF()
    t = 0
    ekf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], A=A, R=R, P=P)

    plt.axis([0, 11, -50, 1000])
    plt.grid(True)
    plt.scatter(t+1, Z[0,t], c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], ekf.X))

    plt.show()

    ## loop overvg bf 
    for t in range(1, X_true.shape[1]):
        plt.grid(True)
        plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(ekf.G))
        #print(t, 'G', ekf.G)
        plt.scatter(t+1, int(Z[0,t]),c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        ekf.predict()

        # update State
        ekf.update(Z[:1,t:t+1])
        plt.scatter(t+1, int(ekf.X[0,0]),c='r')
        str_p.append("[X+: %s]"%(ekf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
        keyboardClick = False
        while keyboardClick != True:
            time.sleep(0.05)
            keyboardClick=plt.waitforbuttonpress()


def example2():
    # time t = 0 - 100
    T = 100
    #### Data simulation #### (below)
    ## Parameters ##
    # Sensor error
    R = np.array([[100]])
    # Prediction transition
    A = np.array([[0.95]])
    # Control transition
    B = np.array([[0.5]])
    # Measurement transition
    C = np.array([[1]])
    # Prediction Covariance init
    P = np.array([[1]])

    # True state at t = 0
    X_true = np.array([[1000]])
    X = np.zeros((1,T))
    Z = np.zeros((1,T))
    X[0,0] = X_true[0,0]
    # Control Signal (assumption in tutorial)
    U = np.zeros((1,T))
    U[0,:] = np.arange(T)
    # Noise added to measurement (assumption in tutorial)
    V = np.random.randint(-50,50,(1,T))
    Z[0,0] = C * X[0,0] + V[0,0]

    # Observation
    for t in range(1, T):
        _AX = A * X[0,t-1]
        _BU = B * U[0,t]
        X[0,t] = _AX + _BU

        _CX_V = C * X[0,t] + V[0,t]
        Z[0,t] = _CX_V 
    ##### Data simulation ####### (above)
    print("[INFO] Generated Data!")
    ekf = EKF()
    t = 0
    ekf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], U=U[:1,t:t+1],
                A=A, B=B, C=C, R=R, P=P)
    plt.axis([0, T+10, -50, 1000])
    plt.grid(True)
    plt.scatter(t+1, Z[0,t], c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], ekf.X))

    for t in range(1,T):
        plt.grid(True)
        #plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(ekf.G))
        #print(t, 'G', ekf.G)
        plt.scatter(t+1, int(Z[0,t]),c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        ekf.predict(U=U[:1,t:t+1])

        # update State
        ekf.update(Z[:1,t:t+1])
        plt.scatter(t+1, int(ekf.X[0,0]),c='r')
        str_p.append("[X+: %s]"%(ekf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
        keyboardClick = False
        while (keyboardClick != True) and ((t%50==0) or (t==T-1)):
            time.sleep(0.05)
            keyboardClick=plt.waitforbuttonpress()

def argparse_get():
    args = argparse.ArgumentParser()
    args.add_argument("-e", type=int, default=0, help="select number of example 0 - 1")
    return args.parse_args()

def main():
    args = argparse_get()
    if args.e == 0:
        example1()
    elif args.e == 1:
        example2()
    #example2()

if __name__ == "__main__":
    main()
