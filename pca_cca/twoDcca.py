import numpy as np
import scipy.linalg as la

def OMat(x, p):
    y = np.random.randn(x, p)
    return la.qr(y,  mode='economic')[0]

def normalize(M, A):
    (U, D, V) = la.svd(M.T@A@M)
    return M@U@np.diag(D**(-1/2))@V

def towDcca_transform(X, L, R):
    (_, k1) = L.shape
    (_, k2) = R.shape
    return (np.matmul(np.matmul(L.T, X), R)).reshape(-1, k1*k2)

def SVRG(u, v, X, Y, rx, M, m, eta):
    (n, _) = X.shape
    for j in range(M):
        w = u.copy()
        batch_grad = X.T@(X@u-Y@v)/n+rx*u
        for t in range(m):
            i = np.random.choice(n, 1)[0]
            grad = (X[i, :].dot(w-u))*X[i:i+1, :].T + rx*(w-u) + batch_grad
            w -= eta*grad
        u = w.copy()
    return u

def oneDcca_oneStep_SVRG(u0, v0, X, Y, M, m, eta, x_regular=1e-4, y_regular=1e-4):
    (n, p) = X.shape
    (_, q) = Y.shape

    Sxx = np.matmul(X.T, X)/n+x_regular*np.eye(p)
    Syy = np.matmul(Y.T, Y)/n+y_regular*np.eye(q)
    
    u = SVRG(u0, v0, X, Y, x_regular, M, m, eta)
    u = normalize(u, Sxx)
    v = SVRG(v0, u, Y, X, y_regular, M, m, eta)
    v = normalize(v, Syy)
    
    du = la.norm(u-u0,ord='fro')
    dv = la.norm(v-v0,ord='fro')
    
    return u, v, du, dv

def twoDcca_mat(X, Y, p1, p2, M, m, eta, loading_init=None, x_regular=1e-4, y_regular=1e-4, iter_max=10):
    (_, mx, nx) = X.shape
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
        (Lx, Ly, dLx, dLy) = oneDcca_oneStep_SVRG(Lx, Ly, XR, YR, M, m, eta, x_regular, y_regular)
        
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry, dRx, dRy) = oneDcca_oneStep_SVRG(Rx, Ry, XL, YL, M, m, eta, x_regular, y_regular)
        
        X_tf = towDcca_transform(X, Lx, Rx)
        Y_tf = towDcca_transform(Y, Ly, Ry)
        corr[n+1] = la.norm(X_tf - Y_tf , ord='fro')
        
        loading = [Lx, Rx, Ly, Ry]
        error[n] = dLx+dLy+dRx+dRy
        
    return loading, corr, error