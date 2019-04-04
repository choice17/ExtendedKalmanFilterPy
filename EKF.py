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
		"_X"
		)
	def __init__(self):
		self._X = None
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
			P_t  = A * P_t-1 * A + Q_t
	Update:
			G_t = ( P_t * C.T ) / ( ( C * P_t * C.T) + R)
			X_t = X_t + G_t * (Zt - C * X_t)
			P_t = ( I - G_t * C ) * P_t

	@param X [N x 1] State vector (True state) 
	@param Z [N x 1] Observation (Measurements)
	@param U [N x 1] Control vector
	@param V [N x 1] Measure Noise
	@param A [N x N] Knownledge transition coef of State vector
	@param B [N x N] Knownledge transition coef of Control vector
	@param C [N x N] Knownledge transition coef of Observation ( Measurement )
	@param P [N x N] Covariance of Prediction
	@param G [N x N] Kalmam gain
	@param R [N x N] Sensor noise
	@param Q [N x N] Process noise
	"""
	def setInit(self, X, Z, U=None, V=None, A=None, B=None, C=None, P=None, G=None, R=None):

		if len(X.shape) == 1:
			X = np.expand_dim(X, axis=1)
		if len(X.shape) == 2:
			self._stateDim = X.shape[0]
		global I
		I = np.eye(self._stateDim)
		self.X = X
		self.Z = Z
		self.U = np.zeros((self._stateDim, self._stateDim)) if U is None else U
		self.V = np.zeros((self._stateDim, self._stateDim)) if V is None else V
		self.A = np.eye(self._stateDim) if A is None else A
		self.B = np.eye(self._stateDim) if B is None else B
		self.C = np.eye(self._stateDim) if C is None else C
		self.P = np.eye(self._stateDim) if P is None else P
		self.G = np.zeros((self._stateDim, self._stateDim)) if G is None else G
		self.R = np.eye(self._stateDim) if R is None else R

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
		self.P = _APAt

	def update(self, Z):
		assert Z.shape == self.Z.shape, "input dim Z is not like previous"
		# Update by observation with Kalman Gain
		self.Z = Z
		_Z_CX = self.Z - np.dot(self.C, self.X)
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