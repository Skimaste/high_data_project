import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def nr_logistic(X, y, maxit=25, tol=1e-10):
    beta = np.zeros(X.shape[1], dtype=float)

    for _ in range(maxit):
        beta_old = beta.copy()
        p = sigmoid(X @ beta)
        w = p * (1.0 - p)
        xtwx = X.T @ (w[:, None] * X)
        score = X.T @ (y - p)
        try:
            delta = np.linalg.solve(xtwx, score)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(xtwx, score, rcond=None)
        beta = beta + delta

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def nr_logistic_ridge(X, y, lambda_, maxit=25, tol=1e-10):
    lambda_ = float(lambda_)
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.eye(X.shape[1], dtype=float) * lambda_
    penalty[0, 0] = 0.0

    for _ in range(maxit):
        beta_old = beta.copy()
        p = sigmoid(X @ beta)
        w = p * (1.0 - p)
        xtwx = X.T @ (w[:, None] * X) + 2.0 * penalty
        score = X.T @ (p - y) + 2.0 * (penalty @ beta)
        try:
            delta = np.linalg.solve(xtwx, score)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(xtwx, score, rcond=None)
        beta = beta - delta

        if np.max(np.abs(beta - beta_old)) < tol:
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
            p = sigmoid(X @ beta_old_nr)
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

            if np.max(np.abs(beta_nr - beta_old_nr)) < tol:
                break

        beta = beta_nr
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    beta[np.abs(beta) <= tol] = 0.0
    return beta


def nll(beta, X, y):
    eta = X @ beta
    p = np.clip(sigmoid(eta), 1e-15, 1.0 - 1e-15)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def predict_proba(beta, X):
    return sigmoid(X @ beta)


def evaluate_model(beta, X, y, threshold=0.5):
    y_prob = predict_proba(beta, X)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }


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
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig("corr_plot.png", dpi=300)
    plt.close()

    random_state = 43
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    cv_ridge_lambda_seq = np.logspace(-1, 1)
    best_lambda_ridge, cv_errors_ridge = cv_ridge(
        X_train_scaled,
        y_train,
        cv_ridge_lambda_seq,
        k=5,
        maxit=25,
        tol=1e-10,
        seed=random_state,
    )
    print("Best Ridge Lambda:", best_lambda_ridge)

    #plot cv errors for ridge
    plt.figure(figsize=(8, 6))
    plt.plot(cv_ridge_lambda_seq, cv_errors_ridge, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Cross-Validation Error')
    plt.title('Ridge Regression Cross-Validation Error')
    plt.grid()
    plt.tight_layout()
    plt.savefig("ridge_cv.png", dpi=300)
    plt.close()


    cv_lasso_lambda_seq = np.logspace(-1, 1)
    best_lambda_lasso, cv_errors_lasso = cv_lasso(
        X_train_scaled,
        y_train,
        cv_lasso_lambda_seq,
        k=5,
        maxitMM=25,
        maxitNR=25,
        tol=1e-10,
        eps=1e-12,
        seed=random_state,
    )
    print("Best Lasso Lambda:", best_lambda_lasso) 

    #plot cv errors for lasso
    plt.figure(figsize=(8, 6))
    plt.plot(cv_lasso_lambda_seq, cv_errors_lasso, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Cross-Validation Error')
    plt.title('Lasso Regression Cross-Validation Error')
    plt.grid()
    plt.tight_layout()
    plt.savefig("lasso_cv.png", dpi=300)
    plt.close()
    

    beta_lr = nr_logistic(X_train_scaled, y_train)
    beta_ridge = nr_logistic_ridge(X_train_scaled, y_train, best_lambda_ridge)
    beta_lasso = nr_logistic_lasso(X_train_scaled, y_train, best_lambda_lasso)

    metrics_logit = evaluate_model(beta_lr, X_test_scaled, y_test)
    metrics_ridge = evaluate_model(beta_ridge, X_test_scaled, y_test)
    metrics_lasso = evaluate_model(beta_lasso, X_test_scaled, y_test)

    results_df = pd.DataFrame(
        [
            {"model": "Logistic (unpenalized)", **metrics_logit},
            {"model": "Ridge Logistic", **metrics_ridge},
            {"model": "Lasso Logistic", **metrics_lasso},
        ]
    )

    print("\nTest set performance:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    coef_df = pd.DataFrame(
        {
            "feature": features,
            "logistic_coef": beta_lr,
            "ridge_coef": beta_ridge,
            "lasso_coef": beta_lasso,
            "abs_lasso_coef": np.abs(beta_lasso),
        }
    ).sort_values("abs_lasso_coef", ascending=False)

    print("\nTop 10 predictors by absolute lasso coefficient:")
    print(
        coef_df.loc[:, ["feature", "lasso_coef", "ridge_coef", "logistic_coef"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    results_df.to_csv("model_performance.csv", index=False)
    coef_df.to_csv("model_coefficients.csv", index=False)
