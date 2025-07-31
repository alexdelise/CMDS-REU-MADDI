import numpy as np

def compute_optimal_mapping(X, Y, F, r, case, reg, affine=False):
    """
    Computes the optimal mapping A_hat (and b_hat if affine=True),
    that minimizes E[ ||AX - Y||^2 ] or E[ ||AY - X||^2 ] depending on the case.

    Parameters:
        X, Y : np.ndarray of shape [params, num_samples]
        F : known forward map
        r : rank constraint (int)
        case : 'forward' or 'inverse'
        reg : regularization term for stability
        affine : if True, includes bias term b

    Returns:
        (A, b) : tuple of optimal linear mapping and bias (None if affine=False)
    """
    params = X.shape[0]

    if affine:
        mu_x = np.mean(X, axis=1, keepdims=True)
        Sigma_X = np.cov(X)
    else:
        mu_x = None
        Sigma_X = X @ X.T / X.shape[1]

    try:
        Sigma_X += reg * np.eye(params)
        L_x = np.linalg.cholesky(Sigma_X)
    except Exception as e:
        raise ValueError(f"Cholesky failed for Sigma_X. Try a larger reg. Details: {e}")

    A = np.zeros((params, params))
    b = None

    if case == 'forward':
        term_1 = F @ L_x
        U, S, Vt = np.linalg.svd(term_1, full_matrices=False)
        r_trunc = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        A = r_trunc @ np.linalg.pinv(L_x)
        if affine:
            b = (F - A) @ mu_x

    elif case == 'inverse':
        if affine:
            Sigma_Y = np.cov(Y)
        else:
            Sigma_Y = Y @ Y.T / Y.shape[1]
        try:
            Sigma_Y += reg * np.eye(params)
            L_y = np.linalg.cholesky(Sigma_Y)
        except Exception as e:
            raise ValueError(f"Cholesky failed for Sigma_Y. Try a larger reg. Details: {e}")

        term_1 = Sigma_X @ F.T @ np.linalg.inv(L_y.T)
        U, S, Vt = np.linalg.svd(term_1, full_matrices=False)
        r_trunc = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        A = r_trunc @ np.linalg.inv(L_y)
        if affine:
            b = (np.eye(params) - A @ F) @ mu_x

    else:
        raise ValueError("`case` must be 'forward' or 'inverse'.")

    return A, b


def compute_reconstruction_error(X, Y, case, optimal_mappings):
    """
    Computes prediction and relative Frobenius error.

    Parameters:
        X, Y : np.ndarray [params, num_samples]
        case : 'forward' or 'inverse'
        optimal_mappings : tuple (A, b) from compute_optimal_mapping

    Returns:
        pred : predicted output
        err : relative Frobenius norm RMSE
    """
    A, b = optimal_mappings
    affine = b is not None

    if case == 'forward':
        pred = A @ X
        if affine:
            pred += b
        err = np.linalg.norm(pred - Y, 'fro') / np.linalg.norm(Y, 'fro')

    elif case == 'inverse':
        pred = A @ Y
        if affine:
            pred += b
        err = np.linalg.norm(pred - X, 'fro') / np.linalg.norm(X, 'fro')

    else:
        raise ValueError("`case` must be 'forward' or 'inverse'.")

    return pred, err

    
    
        







