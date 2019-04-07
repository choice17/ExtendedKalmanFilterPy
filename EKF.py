"""
https://stackoverflow.com/questions/15582956/how-to-pause-a-pylab-figure-until-a-key-is-pressed-or-mouse-is-clicked
"""

import numpy as np

I = None

class KF_interface(object):
	__slots__ = (
		## Kalman Filter Params ##
		'X','Z','U','V',
		'A','B','C','R',
		'P','G',
		## internal ##
		'_stateDim',
		'_sensDim'
		)
	def __init__(self):
		self._stateDim = None
		self._sensDim = None

	def setInit(self, X, Z, U, V, A, B, C, R):
		pass

	def predict(self, U):
		pass

	def update(self):
		pass

class KF(KF_interface):
	"""
	@brief Linear Kalman Filter
	Here is a kalman filter refer to http://home.wlu.edu/~levys/kalman_tutorial/
	
	Use of kalman filter,
	1. Initialize state vector, and uncertainty in measurement and prediction.
	2. Predict X-_t based on our knownledge to the state representation from X+_t-1
	3. Update X+_t based on measurement and the kalman gain.
	   -> also recalculate the kalman gain at this point by comparing prediction error

	State Equation ( Estimation ) :
			X_t = A * X_t-1 + B * U_t 
	Observation ( Real Measurement ) :
			Z_t = C * X_t + V_t

	Predict:
			X_t = A * X_t-1 + B * U_t
			P_t  = A * P_t-1 * A.T + Q_t
	Update:
			G_t = ( P_t * C.T ) / ( ( C * P_t * C.T) + R)
			X_t = X_t + G_t * (Zt - C * X_t)
			P_t = ( I - G_t * C ) * P_t

	State dimension N
	Sensor(observation) dimension M

	@param X [N x 1] State vector (True state) 
	@param Z [M x 1] Observation (Measurements)
	@param U [N x 1] Control vector
	@param A [N x N] Knownledge transition coef of State vector
	@param B [N x N] Knownledge transition coef of Control vector
	@param C [M x N] Knownledge transition coef of Observation ( Measurement )
	@param P [N x N] Covariance of Prediction
	@param G [N x M] Kalmam gain
	@param R [M x M] Sensor noise
	@param Q [N x N] Process noise
	"""
	def setInit(self, X, Z, U=None, A=None, B=None, C=None, P=None, G=None, R=None, Q=None):
		if len(X.shape) == 1:
			X = np.expand_dim(X, axis=1)
		if len(X.shape) == 2:
			self._stateDim = X.shape[0]
		if len(Z.shape) == 1:
			Z = np.expand_dim(Z, axis=1)
		if len(Z.shape) == 2:
			self._sensDim = Z.shape[0] 
		global I
		I = np.eye(self._stateDim)
		self.X = X
		self.Z = Z
		self.U = np.zeros((self._stateDim, 1)) if U is None else U
		self.A = np.eye(self._stateDim) if A is None else A
		self.B = np.eye(self._stateDim) if B is None else B
		if self._stateDim == self._sensDim:
			self.C = np.eye(self._stateDim) if C is None else C
		else:
			if C is None:
				self.C = np.zeros((self._sensDim, self._stateDim))
				print("[WARNING] Please assign sensor contribution for Matrix C")
			else:
				self.C = C
		self.P = np.eye(self._stateDim) if P is None else P
		self.G = np.zeros((self._stateDim, self._sensDim)) if G is None else G
		self.R = np.eye(self._sensDim) if R is None else R
		self.Q = np.zeros((self._stateDim, self._stateDim)) if Q is None else Q

	def predict(self, U=None):
		# Update Prediction
		if U is not None:
			assert U.shape == self.U.shape, "input dim is not correct"
			self.U = U
		_AX = np.dot(self.A, self.X)
		_BU = np.dot(self.B, self.U)
		self.X = _AX + _BU

		# Update Prediction Covariance
		_APAt = np.dot(np.dot(self.A, self.P), self.A.T)
		_APAt_Q = _APAt + self.Q
		self.P = _APAt_Q

	def update(self, Z):
		assert Z.shape == self.Z.shape, "input dim Z is not like previous"
		# Update Kalman Gain
		_PC = np.dot(self.P, self.C.T)
		_CPC_R = np.dot(self.C, _PC) + self.R
		try:
			_CRC_R_inv = np.linalg.inv(_CPC_R)
			self.G = np.dot(_PC, _CRC_R_inv)
		except np.linalg.LinAlgError:
			# Not invertible.
			# Use old kalman gain instead
			print("[Warning] _CRC_R is non invertible ", _CRC_R_inv)
			pass

		# Update by observation with Kalman Gain
		self.Z = Z
		_CX = np.dot(self.C, self.X)
		_Z_CX = self.Z - _CX
		self.X = self.X + np.dot(self.G, _Z_CX)
		
		# Update Prediction Error Covariance after measurement
		_I_GC = I - np.dot(self.G, self.C)
		self.P = np.dot(_I_GC, self.P)



class EKF_interface(object):
	__slots__ = (
		## True Extended Kalman Filter Params ##
		'X','Z','U','V',
		'R','P','G',
		## internal ##
		'_F','_H',
		'_stateDim',
		'_sensDim'
		)
	def __init__(self, stateDim=None, sensDim=None):
		self._stateDim = stateDim
		self._sensDim = sensDim

	def setInit(self, X, Z, U, V, R):
		pass

	def predict(self, U):
		pass

	def update(self):
		pass

	def f(self,X,U):
		pass

	def h(self,X):
		pass

class EKF(EKF_interface):
	"""
	@brief Extended Kalman Filter for non-linear state/observation transition
	Here is a kalman filter refer to http://home.wlu.edu/~levys/kalman_tutorial/
	
	Use of kalman filter,
	1. Initialize state vector, and uncertainty in measurement and prediction.
	2. Predict X-_t based on our knownledge to the state representation from X+_t-1
	3. Update X+_t based on measurement and the kalman gain.
	   -> also recalculate the kalman gain at this point by comparing prediction error

	State Equation ( Estimation ) :
			X_t = f(X_t-1, U_t) + W_t 
	Observation ( Real Measurement ) :
			Z_t = h(X_t) + V_t

	Predict:
			X_t = f(X_t-1, U_t)
			P_t  = F_t-1 * P_t-1 * F_t-1.T + Q_t-1
	Update:
			G_t = ( P_t * H_t.T ) / ( ( H_t * P_t * H_t.T) + R)
			X_t = X_t + G_t * ( Zt - h(X_t) )
			P_t = ( I - G_t * H_t ) * P_t

	State dimension N
	Sensor(observation) dimension M

	@param X [N x 1] State vector (True state) 
	@param Z [M x 1] Observation (Measurements)
	@param U [N x 1] Control vector
	@param W [N x 1] Process noise ( same as Q )
	@param V [M x 1] Measure Noise
	@func f non-linear state transition function
	@param F [N x N] Knownledge non-linear state transition Jacobian matrix (partial derivative)
	@func h non-linear observation transition  function
	@param H [M x N] Knownledge non-linear observation transition of Jacobian matrix (partial derivative)
	@param P [N x N] Covariance of Prediction
	@param G [N x M] Kalmam gain
	@param R [M x M] Sensor noise
	@param Q [N x N] Process noise
	"""
	def setInit(self, X, Z, U=None, P=None, G=None, R=None, Q=None):
		if len(X.shape) == 1:
			X = np.expand_dim(X, axis=1)
		if len(X.shape) == 2:
			self._stateDim = X.shape[0]
		if len(Z.shape) == 1:
			Z = np.expand_dim(Z, axis=1)
		if len(Z.shape) == 2:
			self._sensDim = Z.shape[0] 
		global I
		I = np.eye(self._stateDim)
		self.X = X
		self.Z = Z
		self.U = np.zeros((self._stateDim, 1)) if U is None else U
		self.P = np.eye(self._stateDim) if P is None else P
		self.G = np.zeros((self._stateDim, self._sensDim)) if G is None else G
		self.R = np.eye(self._sensDim) if R is None else R
		self.Q = np.zeros((self._stateDim, self._stateDim)) if Q is None else Q

	def f(self, X, U):
		"""
		Default transition: Linear transformation of X
		Jacobian matrix of X (Partial derivative)
		Advanced usage should include 
		return non-linear transformation of X and the X
		Jacobian matrix of X (Partial derivative) F
		"""
		X, F = X.copy() + U, np.eye(self._stateDim) 
		return X, F

	def h(self, X):
		"""
		Default transition: Linear transformation
		"""
		return X, np.ones((self._sensDim, self._stateDim))

	def predict(self, U=None):
		# Update Prediction
		if U is not None:
			assert U.shape == self.U.shape, "input dim is not correct"
			self.U = U
		self.X, self._F = self.f(self.X, self.U)

		# Update Prediction Covariance
		_FPFt = np.dot(np.dot(self._F, self.P), self._F.T)
		_FPFt_Q = _FPFt + self.Q
		self.P = _FPFt_Q

	def update(self, Z):
		assert Z.shape == self.Z.shape, "input dim Z is not like previous"

		# Update by observation with Kalman Gain
		self.Z = Z
		self._Z, self._H = self.h(self.X)
		self.X = self.X + np.dot(self.G, self.Z - self._Z)

		# Update Kalman Gain
		_PHt = np.dot(self.P, self._H.T)
		_HPHt_R = np.dot(self._H, _PHt) + self.R
		try:
			_HPHt_R_inv = np.linalg.inv(_HPHt_R)
			self.G = np.dot(_PHt, _HPHt_R_inv)
		except np.linalg.LinAlgError:
			# Not invertible.
			# Use old kalman gain instead
			print("[Warning] _HPHt_R is non invertible ", _HPHt_R)
			pass
		
		# Update Prediction Error Covariance after measurement
		_I_GH = I - np.dot(self.G, self._H)
		self.P = np.dot(_I_GH, self.P)