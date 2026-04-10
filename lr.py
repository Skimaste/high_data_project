import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def nr_logistic(X, y, maxit=25, tol=1e-10):
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(maxit):
        beta_old = beta.copy()
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1.0 - p)
        xtwx = X.T @ (w[:, None] * X)
        score = X.T @ (y - p)
        try:
            delta = np.linalg.solve(xtwx, score)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(xtwx, score, rcond=None)
        beta = beta + delta

        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def nr_logistic_ridge(X, y, lambda_, maxit=25, tol=1e-10):
    lambda_ = float(lambda_)
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.eye(X.shape[1], dtype=float) * lambda_
    penalty[0, 0] = 0.0

    for _ in range(maxit):
        beta_old = beta.copy()
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1.0 - p)
        xtwx = X.T @ (w[:, None] * X) + 2.0 * penalty
        score = X.T @ (p - y) + 2.0 * (penalty @ beta)
        try:
            delta = np.linalg.solve(xtwx, score)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(xtwx, score, rcond=None)
        beta = beta - delta

        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def nr_logistic_lasso(X, y, lambda_, maxitMM=25, maxitNR=25, tol=1e-10, eps=1e-12):
    lambda_ = float(lambda_)
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(maxitMM):
        beta_old = beta + np.where(beta > 0.0, eps, -eps)
        beta_nr = beta_old.copy()

        for _ in range(maxitNR):
            beta_old_nr = beta_nr.copy()
            p = 1.0 / (1.0 + np.exp(-(X @ beta_old_nr)))
            w = p * (1.0 - p)

            lam = lambda_ / np.abs(beta_old)
            hessian_partial = np.diag(lam)
            hessian_partial[0, 0] = 0.0

            hessian = X.T @ (w[:, None] * X) + hessian_partial

            score_partial = lam * beta_nr
            score_partial[0] = 0.0

            score = X.T @ (p - y) + score_partial
            try:
                delta = np.linalg.solve(hessian, score)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(hessian, score, rcond=None)
            beta_nr = beta_old_nr - delta

            if np.linalg.norm(beta_nr - beta_old_nr) < tol:
                break

        beta = beta_nr
        if np.linalg.norm(beta - beta_old) < tol:
            break

    beta[np.abs(beta) <= tol] = 0.0
    return beta


def nll(beta, X, y):
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1.0 - 1e-15)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def cv_ridge(X, y, lambda_seq, k=5, maxit=25, tol=1e-10, seed=None):
    lambda_seq = np.asarray(lambda_seq, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.tile(np.arange(k), int(np.ceil(n / k)))[:n]
    rng.shuffle(folds)

    cv_errors = np.zeros(lambda_seq.shape[0], dtype=float)

    for l in range(lambda_seq.shape[0]):
        lambda_ = lambda_seq[l]
        fold_losses = np.zeros(k, dtype=float)

        for fold in range(k):
            idx_valid = np.where(folds == fold)[0]
            idx_train = np.where(folds != fold)[0]

            X_train = X[idx_train]
            y_train = y[idx_train]
            X_valid = X[idx_valid]
            y_valid = y[idx_valid]

            beta_hat = nr_logistic_ridge(X_train, y_train, lambda_, maxit=maxit, tol=tol)
            fold_losses[fold] = nll(beta_hat, X_valid, y_valid)

        cv_errors[l] = np.mean(fold_losses)

    best_idx = np.argmin(cv_errors)
    best_lambda = lambda_seq[best_idx]
    return best_lambda, cv_errors


def cv_lasso(X, y, lambda_seq, k=5, maxitMM=25, maxitNR=25, tol=1e-10, eps=1e-12, seed=None):
    lambda_seq = np.asarray(lambda_seq, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.tile(np.arange(k), int(np.ceil(n / k)))[:n]
    rng.shuffle(folds)

    cv_errors = np.zeros(lambda_seq.shape[0], dtype=float)

    for l in range(lambda_seq.shape[0]):
        lambda_ = lambda_seq[l]
        fold_losses = np.zeros(k, dtype=float)

        for fold in range(k):
            idx_valid = np.where(folds == fold)[0]
            idx_train = np.where(folds != fold)[0]

            X_train = X[idx_train]
            y_train = y[idx_train]
            X_valid = X[idx_valid]
            y_valid = y[idx_valid]

            beta_hat = nr_logistic_lasso(
                X_train,
                y_train,
                lambda_,
                maxitMM=maxitMM,
                maxitNR=maxitNR,
                tol=tol,
                eps=eps,
            )
            fold_losses[fold] = nll(beta_hat, X_valid, y_valid)

        cv_errors[l] = np.mean(fold_losses)

    best_idx = np.argmin(cv_errors)
    best_lambda = lambda_seq[best_idx]
    return best_lambda, cv_errors


if __name__ == "__main__":
    # import kagglehub
    # import os
    # import shutil
    # 
    # Download latest version
    # path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
    # shutil.move(path, os.getcwd())

    data = pd.read_csv('breast-cancer.csv')

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    data = data.drop(columns=['id'])
    X = data[data.columns[1:]].values
    y = data[data.columns[0]].values
    features = data.columns[1:].tolist()

    plt.figure(figsize=(20, 18))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    cv_ridge_lambda_seq = np.logspace(-4, 4, 10)
    best_lambda_ridge, cv_errors_ridge = cv_ridge(X_scaled, y, cv_ridge_lambda_seq, k=5, maxit=25, tol=1e-10, seed=42)
    print("Best Ridge Lambda:", best_lambda_ridge)

    #plot cv errors for ridge
    plt.figure(figsize=(8, 6))
    plt.plot(cv_ridge_lambda_seq, cv_errors_ridge, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Cross-Validation Error')
    plt.title('Ridge Regression Cross-Validation Error')
    plt.grid()
    plt.show()


    cv_lasso_lambda_seq = np.logspace(-4, 4, 10)
    best_lambda_lasso, cv_errors_lasso = cv_lasso(X_scaled, y, cv_lasso_lambda_seq, k=5, maxitMM=25, maxitNR=25, tol=1e-10, eps=1e-12, seed=42)
    print("Best Lasso Lambda:", best_lambda_lasso) 

    #plot cv errors for lasso
    plt.figure(figsize=(8, 6))
    plt.plot(cv_lasso_lambda_seq, cv_errors_lasso, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Cross-Validation Error')
    plt.title('Lasso Regression Cross-Validation Error')
    plt.grid()
    plt.show()
    

    beta_lr = nr_logistic(X_scaled, y)
    print("Coefficients:", np.round(beta_lr, 4))

    beta_ridge = nr_logistic_ridge(X_scaled, y, best_lambda_ridge)
    print("Ridge Coefficients:", np.round(beta_ridge, 4))

    beta_lasso = nr_logistic_lasso(X_scaled, y, best_lambda_lasso)
    print("Lasso Coefficients:", np.array2string(np.round(beta_lasso, 4), formatter={'float_kind': lambda x: f"{x:.4f}"}))
