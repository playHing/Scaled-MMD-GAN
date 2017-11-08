'''
MMD functions implemented in tensorflow.
'''
from __future__ import division

import tensorflow as tf

from tf_ops import dot, sq_sum


_eps=1.0e-5
_check_numerics=False

mysqrt = lambda x: tf.sqrt(tf.maximum(x + _eps, 0.))

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def _distance_kernel(X, Y, K_XY_only=False, check_numerics=_check_numerics):
    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    if check_numerics:
        XX = tf.check_numerics(XX, 'dist XX')
        XY = tf.check_numerics(XY, 'dist XY')
        YY = tf.check_numerics(YY, 'dist YY')
        
    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XY = c(mysqrt(X_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * XY + c(X_sqnorms) + r(Y_sqnorms))

    if check_numerics:
        K_XY = tf.check_numerics(K_XY, 'dist K_XY')
    if K_XY_only:
        return K_XY

    K_XX = c(mysqrt(X_sqnorms)) + r(mysqrt(X_sqnorms)) - mysqrt(-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    K_YY = c(mysqrt(Y_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms))
    if check_numerics:
        K_XX = tf.check_numerics(K_XX, 'dist K_XX')
        K_YY = tf.check_numerics(K_YY, 'dist K_YY')  
        
    return K_XX, K_XY, K_YY, False

def _dot_kernel(X, Y, K_XY_only=False, check_numerics=_check_numerics):
    K_XY = tf.matmul(X, Y, transpose_b=True)
    if check_numerics:
        K_XY = tf.check_numerics(K_XY, 'dot K_XY')
    if K_XY_only:
        return K_XY
    
    K_XX = tf.matmul(X, X, transpose_b=True)
    K_YY = tf.matmul(Y, Y, transpose_b=True)
    if check_numerics:
        K_XX = tf.check_numerics(K_XX, 'dot K_XX')
        K_YY = tf.check_numerics(K_YY, 'dot K_YY')    
    
    return K_XX, K_XY, K_YY, False

def _mix_rbf_kernel(X, Y, sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0], wts=None, 
                    K_XY_only=False, check_numerics=_check_numerics):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    if check_numerics:
        XX = tf.check_numerics(XX, 'rbf XX')
        XY = tf.check_numerics(XY, 'rbf XY')
        YY = tf.check_numerics(YY, 'rbf YY')
        
    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    
    XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XY += wt * tf.exp(-gamma * XYsqnorm)

    if check_numerics:
        K_XY = tf.check_numerics(K_XY, 'rbf K_XY')
        
    if K_XY_only:
        return K_XY
    
    XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
    YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * XXsqnorm)
        K_YY += wt * tf.exp(-gamma * YYsqnorm)

    if check_numerics:
        K_XX = tf.check_numerics(K_XX, 'rbf K_XX')
        K_YY = tf.check_numerics(K_YY, 'rbf K_YY')
        
    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mix_rq_kernel(X, Y, alphas=[.2, .5, .1, 2., 5.], wts=None, 
                   K_XY_only=False, check_numerics=_check_numerics):
    """
    Rational quadratic kernel
    http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    """
    if wts is None:
        wts = [1.] * len(alphas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    if check_numerics:
        XX = tf.check_numerics(XX, 'rq XX')
        XY = tf.check_numerics(XY, 'rq XY')
        YY = tf.check_numerics(YY, 'rq YY')

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)
    
    K_XX, K_XY, K_YY = 0., 0., 0.
    
    XYsqnorm = tf.maximum(-2. * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)
    
    for alpha, wt in zip(alphas, wts):
        logXY = tf.log(1. + XYsqnorm/(2.*alpha))
        if check_numerics:
            logXY = tf.check_numerics(logXY, 'K_XY_log %f' % alpha)
        K_XY += wt * tf.exp(-alpha * logXY)
        
    if check_numerics:
        K_XY = tf.check_numerics(K_XY, 'rq K_XY')

    if K_XY_only:
        return K_XY
    
    XXsqnorm = tf.maximum(-2. * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2. * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)
    
    for alpha, wt in zip(alphas, wts):
        logXX = tf.log(1. + XXsqnorm/(2.*alpha))
        logYY = tf.log(1. + YYsqnorm/(2.*alpha))
        if check_numerics:
            logXX = tf.check_numerics(logXX, 'K_XX_log %f' % alpha)
            logYY = tf.check_numerics(logYY, 'K_YY_log %f' % alpha)
        K_XX += wt * tf.exp(-alpha * logXX)
        K_YY += wt * tf.exp(-alpha * logYY)

    if check_numerics:
        K_XX = tf.check_numerics(K_XX, 'rq K_XX')
        K_YY = tf.check_numerics(K_YY, 'rq K_YY')
    # wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    return K_XX, K_XY, K_YY, wts


def _mix_di_kernel(X, Y, z, alphas, wts=None, K_XY_only=False):
    """
    distance - induced kernel
    k_{alpha,z}(x,x') = d^alpha(x, z) + d^alpha(x', z) - d^alpha(x, x')
    """
    if wts is None:
        wts = [1] * len(alphas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    Xz = tf.matmul(X, z, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    Yz = tf.matmul(Y, z, transpose_b=True)
    zz = tf.matmul(z, z, transpose_b=True)
    
    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)
    z_sqnorms = tf.diag_part(zz)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    d_Xz = c(X_sqnorms) + r(z_sqnorms) - 2 * Xz
    d_Yz = c(Y_sqnorms) + r(z_sqnorms) - 2 * Yz

    K_XX, K_XY, K_YY = 0, 0, 0
    for alpha, wt in zip(alphas, wts):
        p = lambda x: tf.pow(x, alpha)
        K_XY += wt * (p(d_Xz) + p(tf.transpose(d_Yz)) - p(c(X_sqnorms) + r(Y_sqnorms) - 2 * XY))

    if K_XY_only:
        return K_XY

    for alpha, wt in zip(alphas, wts):
        p = lambda x: tf.pow(x, alpha)
        K_XX += wt * (p(d_Xz) + p(tf.transpose(d_Xz)) - p(c(X_sqnorms) + r(X_sqnorms) - 2 * XX))
        K_YY += wt * (p(d_Yz) + p(tf.transpose(d_Yz)) - p(c(Y_sqnorms) + r(Y_sqnorms) - 2 * YY))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def distance_mmd2(X, Y, biased=True):
    K_XX, K_XY, K_YY, d = _distance_kernel(X, Y)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def mix_rq_mmd2(X, Y, alphas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rq_kernel(X, Y, alphas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def mix_di_mmd2(X, Y, z, alphas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_di_kernel(X, Y, z, alphas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

def dot_mmd2(X, Y, biased=True):
    K_XX, K_XY, K_YY = _dot_kernel(X, Y)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def mix_rq_mmd2_and_ratio(X, Y, alphas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rq_kernel(X, Y, alphas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def mix_di_mmd2_and_ratio(X, Y, z, alphas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_di_kernel(X, Y, z, alphas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

def dot_mmd2_and_ratio(X, Y, biased=True):
    K_XX, K_XY, K_YY = _dot_kernel(X, Y)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)    
################################################################################
### Helper functions to compute variances based on kernel matrices


def mmd2(K, biased=False):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased) # numerics checked at _mmd2 return
    
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            const_diagonal = tf.cast(const_diagonal, tf.float32)
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2 #tf.check_numerics(mmd2, '_mmd2 F')#

def mmd2_and_ratio(K, biased=False, min_var_est=_eps):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal, biased, min_var_est)
    
def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est

def _diff_mmd2_and_ratio(K_XY, K_XZ, K_YY, K_ZZ, const_diagonal=False):
    m = tf.cast(K_YY.get_shape()[0], tf.float32)  # Assumes X, Y, Z are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to explicitly form them
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_Y = diag_Z = const_diagonal
#        sum_diag_Y = sum_diag_Z = m * const_diagonal
        sum_diag2_Y = sum_diag2_Z = m * const_diagonal**2    
    else:
        diag_Y = tf.diag_part(K_YY)
        diag_Z = tf.diag_part(K_ZZ)

        sum_diag2_Y = sq_sum(diag_Y)
        sum_diag2_Z = sq_sum(diag_Z)
    
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)

    Kt_ZZ_sums = tf.reduce_sum(K_ZZ, 1) - diag_Z
    Kt_ZZ_sum = tf.reduce_sum(Kt_ZZ_sums)

    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)
    
    K_XZ_sums_0 = tf.reduce_sum(K_XZ, 0)
    K_XZ_sums_1 = tf.reduce_sum(K_XZ, 1)
    
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)
    K_XZ_sum = tf.reduce_sum(K_XZ_sums_0)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    Kt_ZZ_2_sum = sq_sum(K_ZZ) - sum_diag2_Z
    K_XY_2_sum  = sq_sum(K_XY)
    K_XZ_2_sum  = sq_sum(K_XZ)

    ### Estimators for the various terms involved
    muY_muY = Kt_YY_sum / (m * (m-1))
    muZ_muZ = Kt_ZZ_sum / (m * (m-1))

    muX_muY = K_XY_sum / (m * m)
    muX_muZ = K_XZ_sum / (m * m)

    E_y_muY_sq = (sq_sum(Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    E_z_muZ_sq = (sq_sum(Kt_ZZ_sums) - Kt_ZZ_2_sum) / (m*(m-1)*(m-2))

    E_x_muY_sq = (sq_sum(K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    E_x_muZ_sq = (sq_sum(K_XZ_sums_1) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muX_sq = (sq_sum(K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    E_z_muX_sq = (sq_sum(K_XZ_sums_0) - K_XZ_2_sum) / (m*m*(m-1))

    E_y_muY_y_muX = dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_z_muZ_z_muX = dot(Kt_ZZ_sums, K_XZ_sums_0) / (m*m*(m-1))

    E_x_muY_x_muZ = dot(K_XY_sums_1, K_XZ_sums_1) / (m*m*m)

    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kzz2 = Kt_ZZ_2_sum / (m * (m-1))

    E_kxy2 = K_XY_2_sum / (m * m)
    E_kxz2 = K_XZ_2_sum / (m * m)


    ### Combine into overall estimators
    mmd2_diff = muY_muY - 2 * muX_muY - muZ_muZ + 2 * muX_muZ

    first_order = 4 * (m-2) / (m * (m-1)) * (
          E_y_muY_sq - muY_muY**2
        + E_x_muY_sq - muX_muY**2
        + E_y_muX_sq - muX_muY**2
        + E_z_muZ_sq - muZ_muZ**2
        + E_x_muZ_sq - muX_muZ**2
        + E_z_muX_sq - muX_muZ**2
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
        - 2 * E_x_muY_x_muZ + 2 * muX_muY * muX_muZ
        - 2 * E_z_muZ_z_muX + 2 * muZ_muZ * muX_muZ
    )
    second_order = 2 / (m * (m-1)) * (
          E_kyy2 - muY_muY**2
        + 2 * E_kxy2 - 2 * muX_muY**2
        + E_kzz2 - muZ_muZ**2
        + 2 * E_kxz2 - 2 * muX_muZ**2
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muY
        - 4 * E_x_muY_x_muZ + 4 * muX_muY * muX_muZ
        - 4 * E_z_muZ_z_muX + 4 * muZ_muZ * muX_muZ
    )
    var_est = first_order + second_order

    ratio = mmd2_diff / mysqrt(tf.maximum(var_est, _eps))
    return mmd2_diff, ratio