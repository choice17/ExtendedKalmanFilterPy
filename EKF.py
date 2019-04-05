import numpy as np

I = None

class EKF_interface(object):
	__slots__ = (
		## Kalman Filter Params ##
		'X','Z','U','V',
		'A','B','C','R'
		'P','_G',
		## internal ##
		'_stateDim',
		'_sensDim'
		)
	def __init__(self):
		self._stateDim = None

	def setInit(self, X, Z, U, V, A, B, C, R):
		pass

	def predict(self, U):
		pass

	def update(self):
		pass

class EKF(EKF_interface):
	"""
	@brief Extended Kalman Filter
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
	@param V [M x 1] Measure Noise
	@param A [N x N] Knownledge transition coef of State vector
	@param B [N x N] Knownledge transition coef of Control vector
    @param C [M x N] Knownledge transition coef of Observation ( Measurement )
	@param P [N x N] Covariance of Prediction
	@param G [N x M] Kalmam gain
	@param R [M x M] Sensor noise
	@param Q [N x N] Process noise
	"""
	def setInit(self, X, Z, U=None, V=None, A=None, B=None, C=None, P=None, G=None, R=None, Q=None):
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
		self.V = np.zeros((self._stateDim, 1)) if V is None else V
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
		# Update by observation with Kalman Gain
		self.Z = Z
		_CX = np.dot(self.C, self.X)
		_Z_CX = self.Z - _CX
		self.X = self.X + np.dot(self.G, _Z_CX)

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
		
		# Update Prediction Error Covariance after measurement
		_I_GC = I - np.dot(self.G, self.C)
		self.P = np.dot(_I_GC, self.P)