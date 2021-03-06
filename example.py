"""
http://home.wlu.edu/~levys/kalman_tutorial/
"""
import numpy as np
from EKF import KF, EKF
import math
import argparse
import matplotlib.pyplot as plt
import time

def example1_KF():
    """
    Jet landing alttitude
    X_true - true state alttitude in meters
    Z - observation(measurement) alttitude in meters
    """
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
    kf = KF()
    t = 0
    kf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], A=A, R=R, P=P)

    plt.axis([0, 11, -50, 1100])
    plt.grid(True)
    plt.scatter(t+1, Z[0,t], c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], kf.X))

    #plt.show()

    ## loop overvg bf 
    for t in range(1, X_true.shape[1]):
        plt.grid(True)
        plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(kf.G))
        #print(t, 'G', ekf.G)
        plt.scatter(t+1, int(Z[0,t]),c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        kf.predict()

        # update State
        kf.update(Z[:1,t:t+1])
        plt.scatter(t+1, int(kf.X[0,0]),c='r')
        str_p.append("[X+: %s]"%(kf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
        keyboardClick = False
        while keyboardClick != True:
            time.sleep(0.05)
            keyboardClick=plt.waitforbuttonpress()


def example2_KF():
    """
    Jet landing alttitude
    Simulated data for noise V_t
    Set initialize state from 1000
    Generated Data by U, V, X_init, A, B, C
    """

    # time t = 0 - 100
    T = 100
    #### Data simulation #### (below)
    ## Parameters ##
    # Sensor error
    R = np.array([[100]])
    # Prediction transition
    A = np.array([[0.95]])
    # Control transition
    B = np.array([[0.15]])
    # Measurement transition
    C = np.array([[0.5]])
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
    V = np.random.randint(-200,200,(1,T))
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
    kf = KF()
    t = 0
    kf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], U=U[:1,t:t+1],
                A=A, B=B, C=C, R=R, P=P)
    plt.axis([0, T+10, -50, 1100])
    plt.grid(True)
    plt.plot(list(range(T)), X[0,:], color=(0,1,1))
    plt.scatter(t+1, Z[0,t], c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], kf.X))

    for t in range(1,T):
        plt.grid(True)
        #plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(kf.G))
        #print(t, 'G', ekf.G)
        plt.scatter(t+1, int(Z[0,t]),c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        kf.predict(U=U[:1,t:t+1])

        # update State
        kf.update(Z[:1,t:t+1])
        plt.scatter(t+1, int(kf.X[0,0]),c='r')
        str_p.append("[X+: %s]"%(kf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
        keyboardClick = False
        while (keyboardClick != True) and ((t%50==0) or (t==T-1)):
            time.sleep(0.05)
            keyboardClick=plt.waitforbuttonpress()

def example3_KF():
    """
    2.5 sine period
    """
    #### Simulated Data #####
    # Time steps
    T = 500
    tseq = np.linspace(0, 2.5*(2*math.pi), T)
    X = np.zeros((1,T))
    XT = np.zeros((1,T))
    Z = np.zeros((2,T))
    _stateDim = 1
    _sensDim = 2
    # Sensor noise var
    R = np.array([[0.64, 0],
                  [0, 0.64]])
    # Constant transition matrix for State vector
    A = np.eye(_stateDim)
    # Constant transition matrix for Control vector
    B = np.eye(_stateDim)
    # Constant transition matrix for Measurement from State
    C = np.ones((_sensDim,_stateDim))
    # Init prediction err
    P = np.eye(_stateDim)
    # Constant processing noise for both sensor
    Q = np.array([[0.5]])
    # Bias for sensor
    V = np.array([[-1],
                  [1]])
    # Control 
    U = np.zeros((_stateDim,1))

    for t in range(T):
        XT[0,t] = math.sin(tseq[t]) + 20
        X[0,t] = XT[0,t] + np.random.randn(1)*Q
        Z[0,t] = V[0,0] + X[0,t] + np.random.randn(1)*R[0,0]
        Z[1,t] = V[1,0] + X[0,t] + np.random.randn(1)*R[1,1]

    #### Simulated Data above ########
    print("[INFO] Generated Data!")
    kf = KF()
    t = 0
    mean = np.mean(Z[:2,t:t+1])
    _X = np.array([[mean]])
    kf.setInit(X=_X, Z=Z[:2,t:t+1], U=U,
                A=A, B=B, C=C, R=R, P=P, Q=Q)
    plt.axis([0, T+10, 15, 25])
    plt.plot(np.arange(T),X[0,:],np.arange(T),Z[0,:],np.arange(T),Z[1,:])
    plt.grid(True)
    plt.show()
    plt.scatter(t+1, mean, c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], kf.X))
    print("t:%d Z:%s X:%s"%(t, Z[1,t], kf.X))
    update = []
    ti = []
    for t in range(1,T):
        plt.grid(True)
        #plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(kf.G))
        #print(t, 'G', ekf.G)
        #plt.scatter(t+1, Z[0,t],c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        kf.predict()

        # update State
        kf.update(Z=Z[:2,t:t+1])
        ti.append(t+1)
        update.append(kf.X[0,0])
        #plt.scatter(t+1, int(ekf.X[0,0]),c='r')
        #plt.plot(ti, Z[0,0:t], c='b')
        #plt.plot(ti, Z[1,0:t], c='g')
        #plt.plot(ti, update, c='r')
        str_p.append("[X+: %s]"%(kf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
    
    ## 1. Sensor one measurement
    #plt.plot(ti, Z[0,0:t], c='b')
    ## 2. Sensor two measurement
    #plt.plot(ti, Z[1,0:t], c='y')
    ## 3. Mean measurement
    plt.plot(ti, np.mean(Z[:,:t],axis=0), c='b')
    ## 4. Simulated Data
    plt.plot(ti, X[0,0:t], c='g')
    ## 5. Ground True data
    plt.plot(ti, XT[0,0:t], c='c')
    ## 6. Kalman filtered data
    plt.plot(ti, update, c='r')
    keyboardClick = False
    #plt.show()
    while (keyboardClick != True):# and ((t==T-1)):
        time.sleep(0.05)
        keyboardClick=plt.waitforbuttonpress()

def example1_EKF():
    """
    Jet landing alttitude
    X_true - true state alttitude in meters
    Z - observation(measurement) alttitude in meters
    """
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
    def f(X, U): return np.dot(A, X.copy()) , A.copy()
    def h(Z): return Z.copy(), np.eye(ekf._sensDim)

    ekf.f = f
    ekf.h = h

    ekf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], R=R, P=P)


    plt.axis([0, 11, -50, 1100])
    plt.grid(True)
    plt.scatter(t+1, Z[0,t], c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], ekf.X))

    #plt.show()

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


def example2_EKF():
    
    #Jet landing alttitude
    #Simulated data for noise V_t
    #Set initialize state from 1000
    #Generated Data by U, V, X_init, A, B, C
    

    # time t = 0 - 100
    T = 100
    #### Data simulation #### (below)
    ## Parameters ##
    # Sensor error
    R = np.array([[100]])
    # Prediction transition
    A = np.array([[0.95]])
    # Control transition
    B = np.array([[0.15]])
    # Measurement transition
    C = np.array([[0.5]])
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
    V = np.random.randint(-200,200,(1,T))
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
    def f(X, _U):
    	return np.dot(A, X.copy()) + np.dot(B, _U) , A.copy()
    def h(X): return C*X.copy(), np.eye(ekf._sensDim)*C

    ekf.f = f
    ekf.h = h

    #ekf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], R=R, P=P)

    ekf.setInit(X=Z[:1,t:t+1], Z=Z[:1,t:t+1], U=U[:1,t:t+1],
                R=R, P=P)
    plt.axis([0, T+10, -200, 1100])
    plt.grid(True)
    plt.plot(list(range(T)), X[0,:], color=(1,0.5,0))
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



def example3_EKF():
    
    #2.5 sine period
    
    #### Simulated Data #####
    # Time steps
    T = 500
    tseq = np.linspace(0, 2.5*(2*math.pi), T)
    X = np.zeros((1,T))
    XT = np.zeros((1,T))
    Z = np.zeros((2,T))
    _stateDim = 1
    _sensDim = 2
    # Sensor noise var
    R = np.array([[0.64, 0],
                  [0, 0.64]])
    # Constant transition matrix for State vector
    A = np.eye(_stateDim)
    # Constant transition matrix for Control vector
    B = np.eye(_stateDim)
    # Constant transition matrix for Measurement from State
    C = np.ones((_sensDim,_stateDim))
    # Init prediction err
    P = np.eye(_stateDim)
    # Constant processing noise for both sensor
    Q = np.array([[0.5]])
    # Bias for sensor
    V = np.array([[-1],
                  [1]])
    # Control 
    U = np.zeros((_stateDim,1))

    for t in range(T):
        XT[0,t] = math.sin(tseq[t]) + 20
        X[0,t] = XT[0,t] + np.random.randn(1)*Q
        Z[0,t] = V[0,0] + X[0,t] + np.random.randn(1)*R[0,0]
        Z[1,t] = V[1,0] + X[0,t] + np.random.randn(1)*R[1,1]

    #### Simulated Data above ########
    print("[INFO] Generated Data!")
    ekf = EKF()

    def f(_X, _U):
    	return _X.copy(), np.eye(_stateDim)

    def h(_Z):
    	return _Z.copy(), np.ones((_sensDim, _stateDim))

    ekf.f = f
    ekf.h = h
    t = 0
    mean = np.mean(Z[:2,t:t+1])
    _X = np.array([[mean]])
    ekf.setInit(X=_X, Z=Z[:2,t:t+1], R=R, P=P, Q=Q)
    plt.axis([0, T+10, 15, 25])
    plt.plot(np.arange(T),X[0,:],np.arange(T),Z[0,:],np.arange(T),Z[1,:])
    plt.grid(True)
    plt.show()
    plt.scatter(t+1, mean, c='g')
    print("t:%d Z:%s X:%s"%(t, Z[0,t], ekf.X))
    print("t:%d Z:%s X:%s"%(t, Z[1,t], ekf.X))
    update = []
    ti = []
    for t in range(1,T):
        plt.grid(True)
        #plt.pause(0.05)
        str_p = ["t:%d"%t]
        str_p.append("[G: %s]"%(ekf.G))
        #print(t, 'G', ekf.G)
        #plt.scatter(t+1, Z[0,t],c='g')
        str_p.append("[Z: %s]"%(Z[0,t]))
        #print(t, 0, int(Z[0,t]), t+1)

        # initialize State
        ekf.predict()

        # update State
        ekf.update(Z=Z[:2,t:t+1])
        ti.append(t+1)
        update.append(ekf.X[0,0])
        #plt.scatter(t+1, int(ekf.X[0,0]),c='r')
        #plt.plot(ti, Z[0,0:t], c='b')
        #plt.plot(ti, Z[1,0:t], c='g')
        #plt.plot(ti, update, c='r')
        str_p.append("[X+: %s]"%(ekf.X[0,0]))
        #print(t, 1, int(ekf.X[0,0]), t+1)
        print(" ".join(str_p))
    
    ## 1. Sensor one measurement
    #plt.plot(ti, Z[0,0:t], c='b')
    ## 2. Sensor two measurement
    #plt.plot(ti, Z[1,0:t], c='y')
    ## 3. Mean measurement
    plt.plot(ti, np.mean(Z[:,:t],axis=0), c='b')
    ## 4. Simulated Data
    plt.plot(ti, X[0,0:t], c='g')
    ## 5. Ground True data
    plt.plot(ti, XT[0,0:t], c='c')
    ## 6. Kalman filtered data
    plt.plot(ti, update, c='r')
    keyboardClick = False
    plt.show()
    while (keyboardClick != True) and ((t==T-1)):
        time.sleep(0.05)
        keyboardClick=plt.waitforbuttonpress()


def argparse_get():
    args = argparse.ArgumentParser()
    args.add_argument("-e", type=int, default=0, help="select number of example 0 - 1")
    return args.parse_args()

def main():
    args = argparse_get()
    if args.e == 0:
        example1_KF()
    elif args.e == 1:
        example2_KF()
    elif args.e == 2:
        example3_KF()
    elif args.e == 3:
        example1_EKF()
    elif args.e == 4:
        example2_EKF()
    elif args.e == 5:
        example3_EKF()


if __name__ == "__main__":

	"""
	Matplotlib keywait example
    #def quit_figure(event):
    #    if event.key == 't':
    #        plt.close(event.canvas.figure)
    #    elif event.key == 'p':
    #        print("hi I am here")

    #plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
	"""
    main()
