import numpy as np
import scipy.linalg as la

def unitVec(x):
    y = np.random.randn(x, 1)
    return y/la.norm(y)

def OMat(x, p):
    y = np.random.randn(x, p)
    return la.qr(y,  mode='economic')[0]

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

def twoDcca_transform(X, L, R):
    (N, _, _) = X.shape
    (_, k1) = L.shape
    (_, k2) = R.shape
    return (np.matmul(np.matmul(L.T, X), R)).reshape(N, k1*k2)

def oneDcca_ssa(X, Y, k, regular=1e-2):
    (N, p) = X.shape
    (_, q) = Y.shape

    Sxx = X.T@X/N
    Sxy = X.T@Y/N
    Syy = Y.T@Y/N
    A = np.block([[np.zeros((p, p)), Sxy], [Sxy.T, np.zeros((q, q))]])
    B = la.block_diag(Sxx, Syy)+regular*np.eye(p+q)
    (Leval, Levec) = la.eigh(A, b=B)
    Cx = Levec[:p, :k]
    Cy = Levec[p:, :k]

    return -Cx, Cy

def twoDcca_oneStep(ui, vi, X, Y, x_regular, y_regular):
    (n, p) = X.shape
    (n, q) = Y.shape

    # compute sample covariance matrices 
    Sxx = np.matmul(X.T, X)/n + x_regular*np.eye(p)
    Syy = np.matmul(Y.T, Y)/n + y_regular*np.eye(q)
    Sxy = np.matmul(X.T, Y)/n
    
    # update
    u = la.solve(Sxx, Sxy@vi, assume_a='sym')
    u = u/la.norm(u)
    v = la.solve(Syy, Sxy.T@u, assume_a='sym')
    v = v/la.norm(v)

    return u, v

def twoDcca(X, Y, loading_init=None, x_regular=1e-2, y_regular=1e-2, iter_max=10):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(unitVec, (mx, nx, my, ny)))
    
    for n in range(iter_max):    
        XR = np.matmul(X, Rx)[:,:,0]
        YR = np.matmul(Y, Ry)[:,:,0]
        (Lx, Ly) = twoDcca_oneStep(Lx, Ly, XR, YR, x_regular, y_regular)
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry) = twoDcca_oneStep(Rx, Ry, XL, YL, x_regular, y_regular)
    
    X_tf = twoDcca_transform(X, Lx, Rx)
    Y_tf = twoDcca_transform(Y, Ly, Ry)
    corr = np.corrcoef(X_tf.T, Y_tf.T)[0][1]

    return (Lx, Rx, Ly, Ry), corr

def effectiveInit(X, Y, regualr=1e-3):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    Xv = X.reshape(N, mx*nx, order='F')
    Yv = Y.reshape(N, my*ny, order='F')
    
    Cx, Cy = oneDcca_ssa(Xv, Yv, 1, regular=regualr)
    
    Umat = Cx.reshape(mx, nx, order='F')
    Vmat = Cy.reshape(my, ny, order='F')
    (UL, Uval, UR) = la.svd(Umat)
    (VL, Vval, VR) = la.svd(Vmat)
    
    return (UL[:,0:1], UR[0:1,:].T, VL[:,0:1], VR[0:1,:].T)


def normalize(M, A):
    (U, D, V) = la.svd(M.T@A@M)
    return M@U@np.diag(D**(-1/2))@V

def twoDcca_mat_oneStep(U0, V0, X, Y, x_regular, y_regular):
    (n, p) = X.shape
    (_, q) = Y.shape

    # compute sample covariance matrices
    Sxx = np.matmul(X.T, X)/n+x_regular*np.eye(p)
    Sxy = np.matmul(X.T, Y)/n
    Syy = np.matmul(Y.T, Y)/n+y_regular*np.eye(q)
    
    # update
    U = la.solve(Sxx,Sxy@V0, assume_a='sym')
    U = normalize(U, Sxx)
    V = la.solve(Syy,Sxy.T@U, assume_a='sym')
    V = normalize(V, Syy)

    return U, V

def twoDcca_mat(X, Y, p1, p2, loading_init=None, x_regular=1e-2, y_regular=1e-2, iter_max=10):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(OMat, (mx, nx, my, ny), (p1, p2, p1, p2)))
        
    for n in range(iter_max):
        
        XR = np.matmul(X, Rx)[:,:,0]
        YR = np.matmul(Y, Ry)[:,:,0]
        (Lx, Ly) = twoDcca_mat_oneStep(Lx, Ly, XR, YR, x_regular, y_regular)
        
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry) = twoDcca_mat_oneStep(Rx, Ry, XL, YL, x_regular, y_regular)
        
    return (Lx, Rx, Ly, Ry)

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

def initialization(X):
    (N, m, n) = X.shape
    L, R  = [np.random.randn(l, 1) for l in [m, n]]
    return L/la.norm(L), R/la.norm(R)

def proj(X, L, R):
    (N, m, n) = X.shape
    t = (L.T@X@R).reshape(-1,1)
    P = (np.eye(N)-t@t.T/(t.T@t))
    return (P@X.transpose(1,0,2)).transpose(1,0,2)

def towDcca_transform(X, L, R):
    (N, _, _) = X.shape
    (_, k1) = L.shape
    (_, k2) = R.shape
    return (np.matmul(np.matmul(L.T, X), R)).reshape(N, k1*k2)

def twoDcca_deflation(X, Y, x_regular=1e-2, y_regular=1e-2, iter_max=10):
    (_, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx1,Rx1) = initialization(X)
    (Lx2,Rx2) = initialization(X)
    (Ly1,Ry1) = initialization(Y)
    (Ly2,Ry2) = initialization(Y)
        
    (Lx1, Rx1, Ly1, Ry1), _= twoDcca(X, Y, (Lx1, Rx1, Ly1, Ry1), x_regular, y_regular)
    
    X_res = proj(X, Lx1, Rx1)
    Y_res = proj(Y, Ly1, Ry1)
    
    (Lx2, Rx2, Ly2, Ry2), _ = twoDcca(X_res, Y_res, (Lx2, Rx2, Ly2, Ry2), x_regular, y_regular)
    
    for n in range(iter_max):
        X_res = proj(X, Lx2, Rx2)
        Y_res = proj(Y, Ly2, Ry2)
        (Lx1, Rx1, Ly1, Ry1), _ = twoDcca(X_res, Y_res, (Lx1, Rx1, Ly1, Ry1), x_regular, y_regular)
    
        X_res = proj(X, Lx1, Rx1)
        Y_res = proj(Y, Ly1, Ry1)
        (Lx2, Rx2, Ly2, Ry2), _ = twoDcca(X_res, Y_res, (Lx2, Rx2, Ly2, Ry2), x_regular, y_regular)

    return (Lx1, Rx1, Ly1, Ry1), (Lx2, Rx2, Ly2, Ry2)