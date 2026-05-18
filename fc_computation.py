import os
import numpy as np
from sklearn.covariance import LedoitWolf, OAS
input_folder = "processed_data_epochs"
output_folder = "fc_matrices"
os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))
for file in files:
    if not file.endswith("_epochs.npy"):
        continue
    base_name = file.replace("_epochs.npy", "")
    print(f"\nProcessing: {base_name}")
    X = np.load(
        os.path.join(input_folder, f"{base_name}_epochs.npy")
    )
    labels = np.load(
        os.path.join(input_folder, f"{base_name}_labels.npy")
    )
    labels = labels - np.min(labels)
    n_trials = X.shape[0]
    scm_matrices = []
    lw_matrices = []
    oas_matrices = []
    valid_labels = []
    for i in range(n_trials):
        trial = X[i]
        scm = np.corrcoef(trial)
        if np.isnan(scm).any():
            continue
        lw_cov = LedoitWolf().fit(trial.T).covariance_
        lw_std = np.sqrt(np.diag(lw_cov))
        lw_corr = lw_cov / np.outer(lw_std, lw_std)
        oas_cov = OAS().fit(trial.T).covariance_
        oas_std = np.sqrt(np.diag(oas_cov))
        oas_corr = oas_cov / np.outer(oas_std, oas_std)
        scm_matrices.append(scm)
        lw_matrices.append(lw_corr)
        oas_matrices.append(oas_corr)
        valid_labels.append(labels[i])
    scm_matrices = np.array(scm_matrices)
    lw_matrices = np.array(lw_matrices)
    oas_matrices = np.array(oas_matrices)
    valid_labels = np.array(valid_labels)
    np.save(
        os.path.join(output_folder, f"{base_name}_SCM.npy"),
        scm_matrices
    )
    np.save(
        os.path.join(output_folder, f"{base_name}_LW.npy"),
        lw_matrices
    )
    np.save(
        os.path.join(output_folder, f"{base_name}_OAS.npy"),
        oas_matrices
    )
    np.save(
        os.path.join(output_folder, f"{base_name}_labels.npy"),
        valid_labels
    )
    print(
        f"Saved: "
        f"SCM {scm_matrices.shape}, "
        f"LW {lw_matrices.shape}, "
        f"OAS {oas_matrices.shape}"
    )
print("\nFC computation DONE.")

