import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import OAS
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_kernels
def load_balanced_data(data_folder, samples_per_class=45):
    """
    Loads epoched data, truncates all trials to the minimum time length,
    then selects the first `samples_per_class` for each class.
    Returns a balanced dataset of size 2 * samples_per_class.
    """
    all_epochs = []
    all_labels = []
    min_time = None
    for file in sorted(os.listdir(data_folder)):
        if file.endswith("_epochs.npy"):
            base = file.replace("_epochs.npy", "")
            epochs = np.load(os.path.join(data_folder, f"{base}_epochs.npy"))
            # epochs shape: (n_trials, n_channels, n_time)
            if min_time is None or epochs.shape[2] < min_time:
                min_time = epochs.shape[2]
            all_epochs.append(epochs)
            labels = np.load(os.path.join(data_folder, f"{base}_labels.npy"))
            all_labels.append(labels)
    X_truncated = []
    y_truncated = []
    for epochs, labels in zip(all_epochs, all_labels):
        X_truncated.append(epochs[:, :, :min_time])   # keep only first min_time points
        y_truncated.append(labels)
    X_all = np.vstack(X_truncated)
    y_all = np.concatenate(y_truncated)
    unique_labels = np.unique(y_all)
    label_map = {unique_labels[0]: 1, unique_labels[1]: 2}
    y_all = np.array([label_map[l] for l in y_all])
    idx1 = np.where(y_all == 1)[0][:samples_per_class]
    idx2 = np.where(y_all == 2)[0][:samples_per_class]
    idx = np.concatenate([idx1, idx2])
    return X_all[idx], y_all[idx]
def compute_fc_matrix_oas(trial):
    """
    Computes a correlation matrix from a single trial using the OAS estimator.
    Input shape: (channels, time)
    """
    oas = OAS().fit(trial.T)          # fit expects (samples, features)
    cov = oas.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)       # convert to correlation
    return corr
def extract_lower_tri(mat):
    """Extract the lower triangular part of a matrix (excluding diagonal)."""
    return mat[np.tril_indices_from(mat, k=-1)]
def project_data(features, method='mds', random_state=42):
    if method == 'mds':
        return MDS(n_components=2, random_state=random_state).fit_transform(features)
    elif method == 'pca':
        return PCA(n_components=2, random_state=random_state).fit_transform(features)
    else:
        raise ValueError("method must be 'mds' or 'pca'")
def biswas_ghosh_test(X, Y, n_perm=1000):
    nx, ny = len(X), len(Y)
    XY = np.vstack([X, Y])
    D = squareform(pdist(XY))
    W = D[:nx, :nx]
    V = D[nx:, nx:]
    B = D[:nx, nx:]
    T_obs = (np.mean(B) - (np.mean(W) + np.mean(V))/2)
    perm_stats = []
    combined = np.vstack([X, Y])
    for _ in range(n_perm):
        perm_idx = np.random.permutation(nx + ny)
        X_perm = combined[perm_idx[:nx]]
        Y_perm = combined[perm_idx[nx:]]
        Dp = squareform(pdist(np.vstack([X_perm, Y_perm])))
        Wp = Dp[:nx, :nx]
        Vp = Dp[nx:, nx:]
        Bp = Dp[:nx, nx:]
        T_perm = (np.mean(Bp) - (np.mean(Wp) + np.mean(Vp))/2)
        perm_stats.append(T_perm)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(T_obs))
    return p_value
def mmd_test(X, Y, kernel='rbf', gamma=None, n_perm=1000):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = pairwise_kernels(X, metric=kernel, gamma=gamma)
    Kyy = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    Kxy = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    m, n = len(X), len(Y)
    mmd_obs = (Kxx.sum() / (m*m) + Kyy.sum() / (n*n) - 2*Kxy.sum() / (m*n))
    perm_stats = []
    combined = np.vstack([X, Y])
    for _ in range(n_perm):
        perm_idx = np.random.permutation(m + n)
        Xp = combined[perm_idx[:m]]
        Yp = combined[perm_idx[m:]]
        Kxxp = pairwise_kernels(Xp, metric=kernel, gamma=gamma)
        Kyyp = pairwise_kernels(Yp, metric=kernel, gamma=gamma)
        Kxyp = pairwise_kernels(Xp, Yp, metric=kernel, gamma=gamma)
        mmd_perm = (Kxxp.sum() / (m*m) + Kyyp.sum() / (n*n) - 2*Kxyp.sum() / (m*n))
        perm_stats.append(mmd_perm)
    p_value = np.mean(perm_stats >= mmd_obs)
    return p_value
def wasserstein_test(X, Y, n_perm=1000):
    from scipy.stats import wasserstein_distance
    w_obs = wasserstein_distance(X.flatten(), Y.flatten())
    perm_stats = []
    combined = np.vstack([X, Y])
    m, n = len(X), len(Y)
    for _ in range(n_perm):
        perm_idx = np.random.permutation(m + n)
        Xp = combined[perm_idx[:m]]
        Yp = combined[perm_idx[m:]]
        w_perm = wasserstein_distance(Xp.flatten(), Yp.flatten())
        perm_stats.append(w_perm)
    p_value = np.mean(perm_stats >= w_obs)
    return p_value
def run_pipeline_oas(data_folder, samples_per_class=45):
    X_raw, y = load_balanced_data(data_folder, samples_per_class)
    print(f"Data loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} channels, {X_raw.shape[2]} time points")
    ecm_features, lec_features = [], []
    for trial in X_raw:
        fc = compute_fc_matrix_oas(trial)          # OAS correlation matrix
        ecm = extract_lower_tri(fc)
        lec = extract_lower_tri(np.log(np.abs(fc) + 1e-12))
        ecm_features.append(ecm)
        lec_features.append(lec)
    ecm_features = np.array(ecm_features)
    lec_features = np.array(lec_features)
    print(f"ECM features shape: {ecm_features.shape}")
    print(f"LEC features shape: {lec_features.shape}")
    scaler_ecm = StandardScaler().fit(ecm_features)
    scaler_lec = StandardScaler().fit(lec_features)
    ecm_norm = scaler_ecm.transform(ecm_features)
    lec_norm = scaler_lec.transform(lec_features)
    ecm_mds = project_data(ecm_norm, method='mds')
    ecm_pca = project_data(ecm_norm, method='pca')
    lec_mds = project_data(lec_norm, method='mds')
    lec_pca = project_data(lec_norm, method='pca')
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    titles = [["MDS", "PCA"], ["MDS", "PCA"]]
    row_labels = ["ECM", "LEC"]
    data_map = [[ecm_mds, ecm_pca], [lec_mds, lec_pca]]
    colors = {1: "#d62728", 2: "#17becf"}
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            X_proj = data_map[i][j]
            for lab in [1, 2]:
                idx = (y == lab)
                ax.scatter(X_proj[idx, 0], X_proj[idx, 1], c=colors[lab], s=30, alpha=0.8, label=f"Class {lab}")
            ax.set_title(titles[i][j], fontsize=10)
            ax.grid(True)
            if j == 0:
                ax.set_ylabel("Dimension 2")
            if i == 1:
                ax.set_xlabel("Dimension 1")
    for i, name in enumerate(row_labels):
        axes[i, 0].annotate(name, xy=(-0.35, 0.5), xycoords='axes fraction', rotation=90,
                            fontsize=12, weight='bold', va='center')
    plt.suptitle("(B) Oracle Approximating Shrinkage (OAS)", fontsize=14)
    plt.tight_layout()
    plt.savefig("figure6_B.png", dpi=150)
    plt.show()
    print("Figure saved as figure6_B.png")
    print("\n--- Hypothesis Test Results (ECM geometry, OAS) ---")
    bg_p = biswas_ghosh_test(ecm_norm[y==1], ecm_norm[y==2])
    mmd_p = mmd_test(ecm_norm[y==1], ecm_norm[y==2])
    wass_p = wasserstein_test(ecm_norm[y==1], ecm_norm[y==2])
    print(f"Biswas-Ghosh p-value: {bg_p:.4f}")
    print(f"MMD p-value: {mmd_p:.4f}")
    print(f"Wasserstein p-value: {wass_p:.4f}")
    print("\n--- Hypothesis Test Results (LEC geometry, OAS) ---")
    bg_p_lec = biswas_ghosh_test(lec_norm[y==1], lec_norm[y==2])
    mmd_p_lec = mmd_test(lec_norm[y==1], lec_norm[y==2])
    wass_p_lec = wasserstein_test(lec_norm[y==1], lec_norm[y==2])
    print(f"Biswas-Ghosh p-value: {bg_p_lec:.4f}")
    print(f"MMD p-value: {mmd_p_lec:.4f}")
    print(f"Wasserstein p-value: {wass_p_lec:.4f}")
    with open("hypothesis_test_results_OAS.txt", "w") as f:
        f.write("Hypothesis Test Results for EEG Data using OAS estimator (45 samples per class)\n")
        f.write("ECM Geometry:\n")
        f.write(f"  Biswas-Ghosh test: p = {bg_p:.4f}\n")
        f.write(f"  MMD test:          p = {mmd_p:.4f}\n")
        f.write(f"  Wasserstein test:  p = {wass_p:.4f}\n\n")
        f.write("LEC Geometry:\n")
        f.write(f"  Biswas-Ghosh test: p = {bg_p_lec:.4f}\n")
        f.write(f"  MMD test:          p = {mmd_p_lec:.4f}\n")
        f.write(f"  Wasserstein test:  p = {wass_p_lec:.4f}\n")
    print("\nResults saved to hypothesis_test_results_OAS.txt")
if __name__ == "__main__":
    run_pipeline_oas("processed_data_epochs", samples_per_class=45)