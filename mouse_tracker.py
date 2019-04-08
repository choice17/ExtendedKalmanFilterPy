from EKF import EKF
import cv2
import time
import numpy as np

class Tracker(EKF):
    def f(self, X, U):
        return X.copy(), np.eye(self._stateDim)

    def h(self, X):
        return X.copy(), np.eye(self._sensDim)

    def step(self, Z):
        self.predict()
        self.update(Z)
        return self.X

MSEC = 100
BLUE = (255, 0 , 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 150, 255)
YELLOW = (0, 255, 255)
V = 0
LINEW= 1

class MousePlayer(object):
    winName = 'Mouse'
    winW = 600
    winH = 400

    def __init__(self):
        self.tx = self.winW//2
        self.ty = self.winH//2
        self._tx = -1
        self._ty = -1
        self.update = np.array([[self.tx],[self.ty]])
        self._update = np.array([[self.tx],[self.ty]])
        self.setupWin()
        self.setupTracker()

    def setupWin(self):
        cv2.namedWindow(self.winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.winName, self.winW, self.winH)
        cv2.setMouseCallback(self.winName, self.run_getMousePos_event)
        self.graph = np.zeros((self.winW, self.winH, 3),dtype=np.uint8)

    def setupTracker(self):
        self.tracker = Tracker()
        X = np.array([[self.tx],[self.ty]])
        _stateDim = 2
        _sensDim = 2
        # Init prediction err
        pval=10; qval=1e-2; rval=0.1
        P = np.eye(_stateDim) * pval
        Q = np.eye(_stateDim) * qval
        R = np.array([[rval,0],
        	          [0,rval]]) 

        self.tracker.setInit(X=X,Z=X,P=P,Q=Q,R=R)

    def run_copyPos(self):
        self._tx = np.copy(self.tx)
        self._ty = np.copy(self.ty)

    def run_step(self):
        self.run_copyPos()
        self._update = self.update
        Z = np.array([[self.tx],[self.ty]])
        self.update = self.tracker.step(Z=Z)

    def run_getMousePos_event(self, event, x, y, flags, param):
        """
        event == cv2.EVENT_LBUTTONDOWN:
        event == cv2.EVENT_LBUTTONUP:
        """
        self.tx = x + np.random.randn(1)[0]*V
        self.ty = y + np.random.randn(1)[0]*V

    def run(self):
        waitKey = 0
        while True:
            if self._tx > 0:
                cv2.line(self.graph, (int(self._tx), int(self._ty)),
                                     (int(self.tx), int(self.ty)), GREEN, LINEW)
                cv2.line(self.graph, (int(self._update[0,0]), int(self._update[1,0])),
                                     (int(self.update[0,0]), int(self.update[1,0])), ORANGE, LINEW)
                cv2.circle(self.graph, (int(self.tx), int(self.ty)), 2, RED, 2, -1)
                cv2.circle(self.graph, (int(self.update[0,0]), int(self.update[1,0])), 2, YELLOW, 2, -1)
                print("G",self.tracker.G)
                cv2.imshow(self.winName, self.graph)
            self.run_step()
            waitKey = cv2.waitKey(MSEC)
            #self.run_copyPos()
            if waitKey == ord('q'):
                break
            if waitKey == ord('r'):
            	self.graph = np.zeros((self.winW, self.winH, 3),dtype=np.uint8)

        cv2.destroyAllWindows()

def main():
    player = MousePlayer()
    player.run()

if __name__ == '__main__':
    main()