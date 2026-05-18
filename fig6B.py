import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import OAS
from scipy.linalg import cholesky, logm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_kernels
def load_balanced_data(data_folder, samples_per_class=45):
    all_epochs = []
    all_labels = []
    min_time = None
    for file in sorted(os.listdir(data_folder)):
        if file.endswith("_epochs.npy"):
            base = file.replace("_epochs.npy", "")
            epochs = np.load(
                os.path.join(data_folder,
                             f"{base}_epochs.npy")
            )
            labels = np.load(
                os.path.join(data_folder,
                             f"{base}_labels.npy")
            )
            if min_time is None or epochs.shape[2] < min_time:
                min_time = epochs.shape[2]
            all_epochs.append(epochs)
            all_labels.append(labels)
    X_truncated = []
    y_truncated = []
    for epochs, labels in zip(all_epochs, all_labels):
        X_truncated.append(
            epochs[:, :, :min_time]
        )
        y_truncated.append(labels)
    X_all = np.vstack(X_truncated)
    y_all = np.concatenate(y_truncated)
    unique_labels = np.unique(y_all)
    label_map = {
        unique_labels[0]: 1,
        unique_labels[1]: 2
    }
    y_all = np.array([label_map[l] for l in y_all])
    idx1 = np.where(y_all == 1)[0][:samples_per_class]
    idx2 = np.where(y_all == 2)[0][:samples_per_class]
    idx = np.concatenate([idx1, idx2])
    return X_all[idx], y_all[idx]

def compute_fc_matrix_oas(trial):
    oas = OAS().fit(trial.T)
    cov = oas.covariance_
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr = np.nan_to_num(corr)

    # ensure SPD
    corr += 1e-6 * np.eye(corr.shape[0])
    return corr
def compute_ecm_feature(corr):
    # Cholesky decomposition
    L = cholesky(corr, lower=True)
    # Frobenius normalization
    L_norm = L / np.linalg.norm(L, ord='fro')
    # lower triangular vector
    feat = L_norm[np.tril_indices_from(L_norm)]
    return feat
def compute_lec_feature(corr):
    # Cholesky decomposition
    L = cholesky(corr, lower=True)
    # Frobenius normalization
    L_norm = L / np.linalg.norm(L, ord='fro')
    # Matrix logarithm
    L_log = logm(L_norm)
    # remove imaginary noise
    L_log = np.real(L_log)
    # lower triangular vector
    feat = L_log[np.tril_indices_from(L_log)]
    return feat
def frechet_mean(features):
    return np.mean(features, axis=0)
def compute_pga(features):
    mean_feat = frechet_mean(features)
    tangent_vectors = features - mean_feat
    U, S, VT = np.linalg.svd(
        tangent_vectors,
        full_matrices=False
    )
    components = VT[:2]
    projected = tangent_vectors @ components.T
    return projected
def compute_mds(features):
    mds = MDS(
        n_components=2,
        random_state=42,
        dissimilarity='euclidean'
    )
    return mds.fit_transform(features)
def biswas_ghosh_test(X, Y, n_perm=1000):
    nx, ny = len(X), len(Y)
    XY = np.vstack([X, Y])
    D = squareform(pdist(XY))
    W = D[:nx, :nx]
    V = D[nx:, nx:]
    B = D[:nx, nx:]
    T_obs = (
        np.mean(B)
        - (np.mean(W) + np.mean(V)) / 2
    )
    perm_stats = []
    combined = np.vstack([X, Y])
    for _ in range(n_perm):
        perm_idx = np.random.permutation(nx + ny)
        Xp = combined[perm_idx[:nx]]
        Yp = combined[perm_idx[nx:]]
        Dp = squareform(
            pdist(np.vstack([Xp, Yp]))
        )
        Wp = Dp[:nx, :nx]
        Vp = Dp[nx:, nx:]
        Bp = Dp[:nx, nx:]
        T_perm = (
            np.mean(Bp)
            - (np.mean(Wp) + np.mean(Vp)) / 2
        )
        perm_stats.append(T_perm)
    p_value = np.mean(
        np.abs(perm_stats) >= np.abs(T_obs)
    )
    return p_value
def mmd_test(X, Y, gamma=None, n_perm=1000):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = pairwise_kernels(
        X,
        metric='rbf',
        gamma=gamma
    )
    Kyy = pairwise_kernels(
        Y,
        metric='rbf',
        gamma=gamma
    )
    Kxy = pairwise_kernels(
        X,
        Y,
        metric='rbf',
        gamma=gamma
    )
    m = len(X)
    n = len(Y)
    mmd_obs = (
        Kxx.sum() / (m * m)
        + Kyy.sum() / (n * n)
        - 2 * Kxy.sum() / (m * n)
    )
    perm_stats = []
    combined = np.vstack([X, Y])
    for _ in range(n_perm):
        perm_idx = np.random.permutation(m + n)
        Xp = combined[perm_idx[:m]]
        Yp = combined[perm_idx[m:]]
        Kxxp = pairwise_kernels(
            Xp,
            metric='rbf',
            gamma=gamma
        )
        Kyyp = pairwise_kernels(
            Yp,
            metric='rbf',
            gamma=gamma
        )
        Kxyp = pairwise_kernels(
            Xp,
            Yp,
            metric='rbf',
            gamma=gamma
        )
        mmd_perm = (
            Kxxp.sum() / (m * m)
            + Kyyp.sum() / (n * n)
            - 2 * Kxyp.sum() / (m * n)
        )
        perm_stats.append(mmd_perm)
    p_value = np.mean(
        np.array(perm_stats) >= mmd_obs
    )
    return p_value
def wasserstein_test(X, Y, n_perm=1000):
    from scipy.stats import wasserstein_distance
    w_obs = wasserstein_distance(
        X.flatten(),
        Y.flatten()
    )
    perm_stats = []
    combined = np.vstack([X, Y])
    m = len(X)
    n = len(Y)
    for _ in range(n_perm):
        perm_idx = np.random.permutation(m + n)
        Xp = combined[perm_idx[:m]]
        Yp = combined[perm_idx[m:]]
        w_perm = wasserstein_distance(
            Xp.flatten(),
            Yp.flatten()
        )
        perm_stats.append(w_perm)
    p_value = np.mean(
        np.array(perm_stats) >= w_obs
    )
    return p_value
def run_pipeline_oas(data_folder,
                     samples_per_class=45):

    X_raw, y = load_balanced_data(
        data_folder,
        samples_per_class
    )
    print(
        f"Loaded data: {X_raw.shape}"
    )
    ecm_features = []
    lec_features = []
    for trial in X_raw:
        corr = compute_fc_matrix_oas(trial)
        ecm_feat = compute_ecm_feature(corr)
        lec_feat = compute_lec_feature(corr)
        ecm_features.append(ecm_feat)
        lec_features.append(lec_feat)
    ecm_features = np.array(ecm_features)
    lec_features = np.array(lec_features)
    print("ECM shape:", ecm_features.shape)
    print("LEC shape:", lec_features.shape)
    ecm_features = np.nan_to_num(ecm_features)
    lec_features = np.nan_to_num(lec_features)
    ecm_features = StandardScaler().fit_transform(
        ecm_features
    )
    lec_features = StandardScaler().fit_transform(
        lec_features
    )
    ecm_mds = compute_mds(ecm_features)
    lec_mds = compute_mds(lec_features)
    ecm_pga = compute_pga(ecm_features)
    lec_pga = compute_pga(lec_features)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 7)
    )
    data_map = [
        [ecm_mds, ecm_pga],
        [lec_mds, lec_pga]
    ]
    titles = [
        ["MDS", "PGA"],
        ["MDS", "PGA"]
    ]
    row_labels = [
        "ECM",
        "LEC"
    ]
    colors = {
        1: "#d62728",
        2: "#17becf"
    }
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            X_proj = data_map[i][j]
            for lab in [1, 2]:
                idx = y == lab
                ax.scatter(
                    X_proj[idx, 0],
                    X_proj[idx, 1],
                    c=colors[lab],
                    s=30,
                    alpha=0.8,
                    label=f"Class {lab}"
                )
            ax.set_title(
                titles[i][j],
                fontsize=10
            )
            ax.grid(True)
            if j == 0:
                ax.set_ylabel("Dimension 2")
            if i == 1:
                ax.set_xlabel("Dimension 1")
    for i, name in enumerate(row_labels):
        axes[i, 0].annotate(
            name,
            xy=(-0.35, 0.5),
            xycoords='axes fraction',
            rotation=90,
            fontsize=12,
            weight='bold',
            va='center'
        )
    plt.suptitle(
        "(B) Oracle Approximating Shrinkage (OAS)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        "figure6_B_corrected.png",
        dpi=150
    )
    plt.show()
    print(
        "\nSaved as figure6_B_corrected.png"
    )
    print("\n--- ECM Geometry Tests ---")
    bg_ecm = biswas_ghosh_test(
        ecm_features[y == 1],
        ecm_features[y == 2]
    )
    mmd_ecm = mmd_test(
        ecm_features[y == 1],
        ecm_features[y == 2]
    )
    wass_ecm = wasserstein_test(
        ecm_features[y == 1],
        ecm_features[y == 2]
    )
    print(
        "Biswas-Ghosh p-value:",
        round(bg_ecm, 4)
    )
    print(
        "MMD p-value:",
        round(mmd_ecm, 4)
    )
    print(
        "Wasserstein p-value:",
        round(wass_ecm, 4)
    )
    print("\n--- LEC Geometry Tests ---")
    bg_lec = biswas_ghosh_test(
        lec_features[y == 1],
        lec_features[y == 2]
    )
    mmd_lec = mmd_test(
        lec_features[y == 1],
        lec_features[y == 2]
    )
    wass_lec = wasserstein_test(
        lec_features[y == 1],
        lec_features[y == 2]
    )
    print(
        "Biswas-Ghosh p-value:",
        round(bg_lec, 4)
    )
    print(
        "MMD p-value:",
        round(mmd_lec, 4)
    )
    print(
        "Wasserstein p-value:",
        round(wass_lec, 4)
    )
    with open(
        "hypothesis_test_results_OAS_corrected.txt",
        "w"
    ) as f:
        f.write(
            "Hypothesis Test Results using OAS\n\n"
        )
        f.write("ECM Geometry\n")
        f.write(
            f"Biswas-Ghosh: {bg_ecm:.4f}\n"
        )
        f.write(
            f"MMD: {mmd_ecm:.4f}\n"
        )
        f.write(
            f"Wasserstein: {wass_ecm:.4f}\n\n"
        )
        f.write("LEC Geometry\n")
        f.write(
            f"Biswas-Ghosh: {bg_lec:.4f}\n"
        )
        f.write(
            f"MMD: {mmd_lec:.4f}\n"
        )
        f.write(
            f"Wasserstein: {wass_lec:.4f}\n"
        )
    print(
        "\nSaved hypothesis_test_results_OAS_corrected.txt"
    )
if __name__ == "__main__":

    run_pipeline_oas(
        "processed_data_epochs",
        samples_per_class=45
    )