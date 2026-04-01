import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_kernels
input_folder = "processed_data_epochs"
MAX_SAMPLES_PER_CLASS = 45
def biswas_ghosh_test(X, Y, n_perm=1000):
    """Biswas-Ghosh two-sample test based on inter-point distances."""
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
    """Maximum Mean Discrepancy test with permutation."""
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
    """Wasserstein distance test with permutation."""
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
X_all = []
labels_all = []
min_time = None
for file in sorted(os.listdir(input_folder)):
    if file.endswith("_epochs.npy"):
        base = file.replace("_epochs.npy", "")
        X = np.load(os.path.join(input_folder, f"{base}_epochs.npy"))
        y = np.load(os.path.join(input_folder, f"{base}_labels.npy"))
        # X shape: (trials, channels, time)
        n_trials, n_channels, n_time = X.shape
        if min_time is None or n_time < min_time:
            min_time = n_time
        X_all.append(X)
        labels_all.append(y)
print(f"Minimum time points across all epochs: {min_time}")
X_truncated = []
labels_truncated = []
for X, y in zip(X_all, labels_all):
    X_cut = X[:, :, :min_time]          # (trials, channels, min_time)
    X_truncated.append(X_cut)
    labels_truncated.append(y)
X_all = np.vstack(X_truncated)
labels_all = np.concatenate(labels_truncated)
print("Original samples:", len(labels_all))
print("Unique labels:", np.unique(labels_all))
unique_labels = np.unique(labels_all)
label_map = {unique_labels[0]: 1, unique_labels[1]: 2}
labels_all = np.array([label_map[l] for l in labels_all])
idx1 = np.where(labels_all == 1)[0][:MAX_SAMPLES_PER_CLASS]
idx2 = np.where(labels_all == 2)[0][:MAX_SAMPLES_PER_CLASS]
idx = np.concatenate([idx1, idx2])
X = X_all[idx]
labels = labels_all[idx]
print("After sampling:", len(labels))
def compute_lw_fc(trial):
    """
    trial: (time, channels)
    return: correlation matrix
    """
    lw = LedoitWolf().fit(trial)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr
fc_matrices = []
for trial in X:
    trial_t = trial.T   # (time, channels)
    fc = compute_lw_fc(trial_t)
    fc_matrices.append(fc)
fc_matrices = np.array(fc_matrices)  # (samples, channels, channels)
def extract_lower(mat):
    return mat[np.tril_indices_from(mat, k=-1)]
ecm_features = np.array([extract_lower(m) for m in fc_matrices])
lec_features = np.array([extract_lower(np.log(np.abs(m) + 1e-12)) for m in fc_matrices])
ecm_features = StandardScaler().fit_transform(ecm_features)
lec_features = StandardScaler().fit_transform(lec_features)
print("\n--- Running hypothesis tests on ECM geometry ---")
bg_ecm = biswas_ghosh_test(ecm_features[labels==1], ecm_features[labels==2])
mmd_ecm = mmd_test(ecm_features[labels==1], ecm_features[labels==2])
wasser_ecm = wasserstein_test(ecm_features[labels==1], ecm_features[labels==2])
print(f"Biswas-Ghosh p-value (ECM): {bg_ecm:.4f}")
print(f"MMD p-value (ECM): {mmd_ecm:.4f}")
print(f"Wasserstein p-value (ECM): {wasser_ecm:.4f}")
print("\n--- Running hypothesis tests on LEC geometry ---")
bg_lec = biswas_ghosh_test(lec_features[labels==1], lec_features[labels==2])
mmd_lec = mmd_test(lec_features[labels==1], lec_features[labels==2])
wasser_lec = wasserstein_test(lec_features[labels==1], lec_features[labels==2])
print(f"Biswas-Ghosh p-value (LEC): {bg_lec:.4f}")
print(f"MMD p-value (LEC): {mmd_lec:.4f}")
print(f"Wasserstein p-value (LEC): {wasser_lec:.4f}")
with open("hypothesis_test_results_LW.txt", "w") as f:
    f.write("Hypothesis Test Results for EEG Data – Ledoit-Wolf Estimator (45 samples per class)\n")
    f.write("ECM Geometry:\n")
    f.write(f"  Biswas-Ghosh test: p = {bg_ecm:.4f}\n")
    f.write(f"  MMD test:          p = {mmd_ecm:.4f}\n")
    f.write(f"  Wasserstein test:  p = {wasser_ecm:.4f}\n\n")
    f.write("LEC Geometry:\n")
    f.write(f"  Biswas-Ghosh test: p = {bg_lec:.4f}\n")
    f.write(f"  MMD test:          p = {mmd_lec:.4f}\n")
    f.write(f"  Wasserstein test:  p = {wasser_lec:.4f}\n")
print("\nResults saved to hypothesis_test_results_LW.txt")
mds = MDS(n_components=2, random_state=42)
ecm_mds = mds.fit_transform(ecm_features)
lec_mds = mds.fit_transform(lec_features)
pca = PCA(n_components=2)
ecm_pga = pca.fit_transform(ecm_features)
lec_pga = pca.fit_transform(lec_features)
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
data_map = [
    [ecm_mds, ecm_pga],
    [lec_mds, lec_pga]
]
titles = [["MDS", "PGA"], ["MDS", "PGA"]]
row_labels = ["ECM", "LEC"]
colors = {1: "#d62728", 2: "#17becf"}
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        X_plot = data_map[i][j]
        for lab in [1, 2]:
            idx_l = labels == lab
            ax.scatter(X_plot[idx_l, 0], X_plot[idx_l, 1],
                       c=colors[lab], s=30, alpha=0.8)
        ax.set_title(titles[i][j], fontsize=10)
        ax.grid(True)
        if j == 0:
            ax.set_ylabel("Dimension 2")
        if i == 1:
            ax.set_xlabel("Dimension 1")
for i, name in enumerate(row_labels):
    axes[i, 0].annotate(name, xy=(-0.35, 0.5), xycoords='axes fraction',
                        rotation=90, fontsize=12, weight='bold', va='center')
plt.suptitle("(A) Ledoit-Wolf (LW)", fontsize=14)
plt.tight_layout()
plt.savefig("fig6_A.png", dpi=150)
plt.show()
print("saved as fig6_A.png")