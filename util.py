import numpy as np
import scipy.linalg as la

def unitVec(x):
    y = np.random.randn(x, 1)
    return y/la.norm(y)

def OMat(x, p):
    y = np.random.randn(x, p)
    return la.qr(y,  mode='economic')[0]

def towDcca_transform(X, L, R):
    (N, _, _) = X.shape
    (_, k1) = L.shape
    (_, k2) = R.shape
    return (np.matmul(np.matmul(L.T, X), R)).reshape(N, k1*k2)

def twoDcca_oneStep(ui, vi, X, Y, x_regular=1e-4, y_regular=1e-4):
    (N, p) = X.shape
    (N, q) = Y.shape

    Sxx = np.matmul(X.T, X)/N
    Sxy = np.matmul(X.T, Y)/N
    Syy = np.matmul(Y.T, Y)/N
    
    u = la.solve((Sxx+x_regular*np.eye(p)),Sxy@vi)
    u /= la.norm(u)
    du = u-ui
    v = la.solve((Syy+y_regular*np.eye(q)),Sxy.T@u)
    v /= la.norm(v)
    dv = v-vi

    return u, v, du.T@du, dv.T@dv

def twoDcca(X, Y, loading_init=None, x_regular=1e-4, y_regular=1e-4, iter_max=10):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(unitVec, (mx, nx, my, ny)))
  
    loading = [Lx, Rx, Ly, Ry]
    
    corr = np.zeros(iter_max+1)
    error = np.zeros(iter_max)
    error_Lx = np.zeros(iter_max)
    error_Rx = np.zeros(iter_max)
    error_Ly = np.zeros(iter_max)
    error_Ry = np.zeros(iter_max)
    
    X_tf = towDcca_transform(X, Lx, Rx)
    Y_tf = towDcca_transform(Y, Ly, Ry)
    corr[0] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]

    for n in range(iter_max):
        
        XR = np.matmul(X, Rx)[:,:,0]
        YR = np.matmul(Y, Ry)[:,:,0]
        (Lx, Ly, dLx, dLy) = twoDcca_oneStep(Lx, Ly, XR, YR, x_regular, y_regular)
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry, dRx, dRy) = twoDcca_oneStep(Rx, Ry, XL, YL, x_regular, y_regular)
        
        X_tf = towDcca_transform(X, Lx, Rx)
        Y_tf = towDcca_transform(Y, Ly, Ry)
      
        loading = [Lx, Rx, Ly, Ry]
        corr[n+1] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]
        error[n] = dLx+dLy+dRx+dRy
        error_Lx[n] = dLx
        error_Rx[n] = dRx
        error_Ly[n] = dLy
        error_Ry[n] = dRy
        
    return loading, corr, (error, error_Lx, error_Rx, error_Ly, error_Ry)
    
def normalize(M, A):
    (U, D, V) = la.svd(M.T@A@M)
    return M@U@np.diag(D**(-1/2))@V

def twoDcca_mat_oneStep(U0, V0, X, Y, x_regular=1e-4, y_regular=1e-4):
    (n, p) = X.shape
    (_, q) = Y.shape

    Sxx = np.matmul(X.T, X)/n+x_regular*np.eye(p)
    Sxy = np.matmul(X.T, Y)/n
    Syy = np.matmul(Y.T, Y)/n+y_regular*np.eye(q)
    
    U = la.solve(Sxx,Sxy@V0)
    U = normalize(U, Sxx)
    dU = la.norm(U-U0,ord='fro')
    
    V = la.solve(Syy,Sxy.T@U)
    V = normalize(V, Syy)
    dV = la.norm(V-V0,ord='fro')

    return U, V, dU, dV

def twoDcca_mat(X, Y, p1, p2, loading_init=None, x_regular=1e-4, y_regular=1e-4, iter_max=10):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    loading = (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(OMat, (mx, nx, my, ny), (p1, p2, p1, p2)))
    
    corr = np.zeros(iter_max+1)
    error = np.zeros(iter_max)

    X_tf = towDcca_transform(X, Lx, Rx)
    Y_tf = towDcca_transform(Y, Ly, Ry)
    corr[0] = la.norm(X_tf - Y_tf , ord='fro')
    
    for n in range(iter_max):
        
        XR = np.matmul(X, Rx)[:,:,0]
        YR = np.matmul(Y, Ry)[:,:,0]
        (Lx, Ly, dLx, dLy) = twoDcca_mat_oneStep(Lx, Ly, XR, YR, x_regular, y_regular)
        
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry, dRx, dRy) = twoDcca_mat_oneStep(Rx, Ry, XL, YL, x_regular, y_regular)
        X_tf = towDcca_transform(X, Lx, Rx)
        Y_tf = towDcca_transform(Y, Ly, Ry)
        corr[n+1] = la.norm(X_tf - Y_tf , ord='fro')
        
        loading = [Lx, Rx, Ly, Ry]
        error[n] = dLx+dLy+dRx+dRy
        
    return loading, corr, error

def twoDpca_oneStep(u, X, rx):
    (n, p) = X.shape
    Sxx = np.matmul(X.T, X)/n+rx*np.eye(p)
    return la.qr(Sxx@u,  mode='economic')[0]

def twoDpca(X, k1, k2, rx, iter_max):
    (_, m, n) = X.shape

    loading = (L,R) = list(map(OMat, (m, n), (k1, k2)))

    for n in range(iter_max):
        XR = np.matmul(X, R)[:,:,0]
        L = twoDpca_oneStep(L, XR, rx)
        
        LX = np.matmul(X.transpose(0,2,1), L)[:,:,0]
        R = twoDpca_oneStep(R, LX, rx)
        
        loading = (L, R)
        
    return loading