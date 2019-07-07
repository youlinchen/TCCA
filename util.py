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

def towDcca_transform(X, L, R):
    (N, _, _) = X.shape
    (_, k1) = L.shape
    (_, k2) = R.shape
    return (np.matmul(np.matmul(L.T, X), R)).reshape(N, k1*k2)

def oneDcca_ssa(X, Y, k, regular=1e-6):
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

def twoDcca_SVD(X, Y, k1, k2, loading_init=None, regular=1e-6, iter_max=10):
    (T, mx, nx) = X.shape
    (T, my, ny) = Y.shape

    corr = np.zeros(iter_max+1)    
    error = np.zeros(iter_max+1)
    
    (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(lambda x: np.random.randn(x, 1), (mx, nx, my, ny)))
    X_tf = towDcca_transform(X, Lx, Rx)
    Y_tf = towDcca_transform(Y, Ly, Ry)
    corr[0] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]
    
    for i in range(iter_max):
        diff = 0
        # estimate Lx, Ly
        XR = np.matmul(X, Rx)
        YR = np.matmul(Y, Ry)
        Srxx = np.matmul(XR, XR.transpose(0,2,1)).sum(axis=0)/T
        Srxy = np.matmul(XR, YR.transpose(0,2,1)).sum(axis=0)/T
        Sryy = np.matmul(YR, YR.transpose(0,2,1)).sum(axis=0)/T
        A = np.block([[np.zeros((mx, mx)), Srxy], [Srxy.T, np.zeros((my, my))]])
        B = la.block_diag(Srxx, Sryy)+regular*np.eye(mx+my)
        (Leval, Levec) = la.eigh(A, b=B)
        diff += la.norm(Lx+Levec[:mx, :k1])
        Lx = -Levec[:mx, :k1]
        diff += la.norm(Ly-Levec[mx:, :k1])
        Ly = Levec[mx:, :k1]

        # estimate Rx, Ry
        XL = np.matmul(X.transpose(0,2,1), Lx)
        YL = np.matmul(Y.transpose(0,2,1), Ly)
        Slxx = np.matmul(XL, XL.transpose(0,2,1)).sum(axis=0)/T
        Slxy = np.matmul(XL, YL.transpose(0,2,1)).sum(axis=0)/T
        Slyy = np.matmul(YL, YL.transpose(0,2,1)).sum(axis=0)/T
        A = np.block([[np.zeros((nx, nx)), Slxy], [Slxy.T, np.zeros((ny, ny))]])
        B = la.block_diag(Slxx, Slyy)+regular*np.eye(nx+ny)
        (Reval, Revec) = la.eigh(A, b=B)
        diff += la.norm(Rx+Revec[:nx, :k2])
        Rx = -Revec[:nx, :k2]
        diff += la.norm(Ry-Revec[nx:, :k2])
        Ry = Revec[nx:, :k2]
        
        X_tf = towDcca_transform(X, Lx, Rx)
        Y_tf = towDcca_transform(Y, Ly, Ry)
        corr[i+1] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]
        error[i+1] = diff
        
    return (Lx, Rx, Ly, Ry), corr, error

def twoDcca_oneStep(ui, vi, X, Y, method, updating_rule, para, x_regular, y_regular):
    (N, p) = X.shape
    (N, q) = Y.shape

    Sxx = np.matmul(X.T, X)/N
    Sxy = np.matmul(X.T, Y)/N
    Syy = np.matmul(Y.T, Y)/N
    
    if updating_rule == 'exact':
        u = la.solve((Sxx+x_regular*np.eye(p)),Sxy@vi)
    if updating_rule == 'inexact':
        M, m, eta = para
        u = SVRG(ui, vi, X, Y, x_regular, M, m, eta)
    if method=='pi':
        u /= np.sqrt(u.T@Sxx@u)
    if method=='als':
        u /= la.norm(u)
    du = u-ui

    if updating_rule == 'exact':
        v = la.solve((Syy+y_regular*np.eye(q)),Sxy.T@u)
    if updating_rule == 'inexact':
        M, m, eta = para
        v = SVRG(vi, u, Y, X, y_regular, M, m, eta)
    if method=='pi':
        v /= np.sqrt(v.T@Syy@v)
    if method=='als':
        v /= la.norm(v)
    dv = v-vi

    return u, v, du.T@du, dv.T@dv

def twoDcca(X, Y, method='als', updating_rule='exact', para=None, loading_init=None, x_regular=1e-4, y_regular=1e-4, iter_max=10):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx, Rx, Ly, Ry) = loading_init if loading_init else list(map(unitVec, (mx, nx, my, ny)))
  
    loading = [Lx, Rx, Ly, Ry]
    
    corr = np.zeros(iter_max+1)
    error = np.zeros(iter_max)
    
    X_tf = towDcca_transform(X, Lx, Rx)
    Y_tf = towDcca_transform(Y, Ly, Ry)
    corr[0] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]

    for n in range(iter_max):
        
        XR = np.matmul(X, Rx)[:,:,0]
        YR = np.matmul(Y, Ry)[:,:,0]
        (Lx, Ly, dLx, dLy) = twoDcca_oneStep(Lx, Ly, XR, YR, method, updating_rule, para, x_regular, y_regular)
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry, dRx, dRy) = twoDcca_oneStep(Rx, Ry, XL, YL, method, updating_rule, para, x_regular, y_regular)
        
        X_tf = towDcca_transform(X, Lx, Rx)
        Y_tf = towDcca_transform(Y, Ly, Ry)
      
        loading = [Lx, Rx, Ly, Ry]
        corr[n+1] = np.corrcoef(X_tf.T, Y_tf.T)[0][1]
        error[n] = dLx+dLy+dRx+dRy
        
    return loading, corr, error
    
def effectiveInit(X, Y, regular=1e-6):
    (N, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    Xv = X.reshape(N, mx*nx, order='F')
    Yv = Y.reshape(N, my*ny, order='F')
    
    Cx, Cy = oneDcca_ssa(Xv, Yv, 1, regular)
    
    Umat = Cx.reshape(mx, nx, order='F')
    Vmat = Cy.reshape(my, ny, order='F')
    (UL, Uval, UR) = la.svd(Umat)
    (VL, Vval, VR) = la.svd(Vmat)
    
    return (UL[:,0:1], UR[0:1,:].T, VL[:,0:1], VR[0:1,:].T)

def findOpt(M, I, X, Y):
    corrOpt = np.zeros(M)
    for m in range(M):
        (loading, corr, error) = twoDcca(X, Y, None, iter_max=I)
        corrOpt[m] = corr[-1]
    return corrOpt.max()

def normalize(M, A):
    (U, D, V) = la.svd(M.T@A@M)
    return M@U@np.diag(D**(-1/2))@V

def twoDcca_mat_oneStep(U0, V0, X, Y, updating_rule, para, x_regular=1e-4, y_regular=1e-4):
    (n, p) = X.shape
    (_, q) = Y.shape

    Sxx = np.matmul(X.T, X)/n+x_regular*np.eye(p)
    Sxy = np.matmul(X.T, Y)/n
    Syy = np.matmul(Y.T, Y)/n+y_regular*np.eye(q)
    
    if updating_rule == 'exact':
        U = la.solve(Sxx,Sxy@V0)
    if updating_rule == 'inexact':
        M, m, eta = para
        U = SVRG(U0, V0, X, Y, x_regular, M, m, eta)
    U = normalize(U, Sxx)
    dU = la.norm(U-U0,ord='fro')
    
    if updating_rule == 'exact':
        V = la.solve(Syy,Sxy.T@U)
    if updating_rule == 'inexact':
        M, m, eta = para
        V = SVRG(V0, U, Y, X, y_regular, M, m, eta)
    
    V = normalize(V, Syy)
    dV = la.norm(V-V0,ord='fro')

    return U, V, dU, dV

def twoDcca_mat(X, Y, p1, p2, loading_init=None, updating_rule='exact', para=None, x_regular=1e-4, y_regular=1e-4, iter_max=10):
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
        (Lx, Ly, dLx, dLy) = twoDcca_mat_oneStep(Lx, Ly, XR, YR, updating_rule, para, x_regular, y_regular)
        
        XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
        YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
        (Rx, Ry, dRx, dRy) = twoDcca_mat_oneStep(Rx, Ry, XL, YL, updating_rule, para, x_regular, y_regular)
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

def oneDcca_oneStep_SVRG(u0, v0, X, Y, M, m, eta, x_regular=1e-4, y_regular=1e-4):
    (n, p) = X.shape
    (_, q) = Y.shape

    Sxx = np.matmul(X.T, X)/n+x_regular*np.eye(p)
    Syy = np.matmul(Y.T, Y)/n+y_regular*np.eye(q)
    
    u = SVRG(u0, v0, X, Y, x_regular, M, m, eta)
    u /= la.norm(u)
    
    v = SVRG(v0, u, Y, X, y_regular, M, m, eta)
    v /= la.norm(v)
    
    du = la.norm(u-u0,ord='fro')
    dv = la.norm(v-v0,ord='fro')
    
    return u, v, du, dv

def initialization(X):
    (N, m, n) = X.shape
    L, R  = [np.random.randn(l, 1) for l in [m, n]]
    return L/la.norm(L), R/la.norm(R)

def proj(X, L, R):
    (N, m, n) = X.shape
    t = (L.T@X@R).reshape(-1,1)
    P = (np.eye(N)-t@t.T/(t.T@t))
    return (P@X.transpose(1,0,2)).transpose(1,0,2)

def twoDcca_iter_onestep(X, Y, Lx, Rx, Ly, Ry, M, m, eta, x_regular, y_regular):
    XR = np.matmul(X, Rx)[:,:,0]
    YR = np.matmul(Y, Ry)[:,:,0]
    (Lx, Ly, dLx, dLy) = oneDcca_oneStep_SVRG(Lx, Ly, XR, YR, M, m, eta, x_regular, y_regular)
        
    XL = np.matmul(X.transpose(0,2,1), Lx)[:,:,0]
    YL = np.matmul(Y.transpose(0,2,1), Ly)[:,:,0]
    (Rx, Ry, dRx, dRy) = oneDcca_oneStep_SVRG(Rx, Ry, XL, YL, M, m, eta, x_regular, y_regular)

    X_tf = towDcca_transform(X, Lx, Rx)
    Y_tf = towDcca_transform(Y, Ly, Ry)

    return (Lx, Rx, Ly, Ry), la.norm(X_tf - Y_tf , ord='fro'), dLx+dLy+dRx+dRy
    
def twoDcca_iter:(X, Y, M, m, eta, x_regular=1e-4, y_regular=1e-4, iter_max=10)
    (_, mx, nx) = X.shape
    (_, my, ny) = Y.shape
    
    (Lx1,Rx1) = initialization(X)
    (Lx2,Rx2) = initialization(X)
    (Ly1,Ry1) = initialization(Y)
    (Ly2,Ry2) = initialization(Y)
    
    residual = np.zeros(iter_max+1)
    error = np.zeros(iter_max+1)
        
    (Lx1, Rx1, Ly1, Ry1), l1, e1 = twoDcca_iter_onestep(X, Y, Lx1, Rx1, Ly1, Ry1, M, m, eta, x_regular, y_regular)
    
    X_res = proj(X, Lx1, Rx1)
    Y_res = proj(Y, Ly1, Ry1)
    
    (Lx2, Rx2, Ly2, Ry2), l2, e2 = twoDcca_iter_onestep(X_res, Y_res, Lx2, Rx2, Ly2, Ry2, M, m, eta, x_regular, y_regular)
    
    residual[0] = l1 + l2
    error[0] = e1 + e2
    
    for n in range(iter_max):
        X_res = proj(X, Lx2, Rx2)
        Y_res = proj(Y, Ly2, Ry2)
        (Lx1, Rx1, Ly1, Ry1), l1, e1 = twoDcca_iter_onestep(X_res, Y_res, Lx1, Rx1, Ly1, Ry1, M, m, eta, x_regular, y_regular)
    
        X_res = proj(X, Lx1, Rx1)
        Y_res = proj(Y, Ly1, Ry1)

        (Lx2, Rx2, Ly2, Ry2), l2, e2 = twoDcca_iter_onestep(X_res, Y_res, Lx2, Rx2, Ly2, Ry2, M, m, eta, x_regular, y_regular)

        residual[0] = l1 + l2
        error[0] = e1 + e2
        
    return (Lx1, Rx1, Ly1, Ry1), (Lx2, Rx2, Ly2, Ry2), residual, error