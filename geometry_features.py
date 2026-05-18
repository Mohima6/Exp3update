import os
import numpy as np
input_folder = "fc_matrices"
output_folder = "geometry_features"
os.makedirs(output_folder, exist_ok=True)
epsilon = 1e-6
def cholesky_normalization(matrix):
    matrix = matrix + epsilon * np.eye(matrix.shape[0])
    L = np.linalg.cholesky(matrix)
    norm_L = L / np.linalg.norm(L)
    return norm_L
def compute_ecm_features(matrix):
    norm_L = cholesky_normalization(matrix)
    idx = np.tril_indices_from(norm_L)
    features = norm_L[idx]
    return features
def compute_lec_features(matrix):
    norm_L = cholesky_normalization(matrix)
    log_L = np.log1p(norm_L)
    idx = np.tril_indices_from(log_L)
    features = log_L[idx]
    return features
files = sorted(os.listdir(input_folder))
estimators = ["SCM", "LW", "OAS"]
for file in files:
    if not file.endswith("_labels.npy"):
        continue
    base_name = file.replace("_labels.npy", "")
    print(f"\nProcessing: {base_name}")
    labels = np.load(
        os.path.join(input_folder, file)
    )
    for estimator in estimators:
        matrix_file = f"{base_name}_{estimator}.npy"
        matrix_path = os.path.join(
            input_folder,
            matrix_file
        )
        if not os.path.exists(matrix_path):
            print(f"Missing: {matrix_file}")
            continue
        matrices = np.load(matrix_path)
        ecm_features = []
        lec_features = []
        for i in range(matrices.shape[0]):
            matrix = matrices[i]
            if np.isnan(matrix).any():
                continue
            try:
                ecm = compute_ecm_features(matrix)
                lec = compute_lec_features(matrix)
                ecm_features.append(ecm)
                lec_features.append(lec)
            except np.linalg.LinAlgError:
                print(
                    f"Cholesky failed for "
                    f"{base_name} | "
                    f"{estimator} | "
                    f"trial {i}"
                )
                continue
        ecm_features = np.array(ecm_features)
        lec_features = np.array(lec_features)
        np.save(
            os.path.join(
                output_folder,
                f"{base_name}_ECM_{estimator}.npy"
            ),
            ecm_features
        )
        np.save(
            os.path.join(
                output_folder,
                f"{base_name}_LEC_{estimator}.npy"
            ),
            lec_features
        )
        np.save(
            os.path.join(
                output_folder,
                f"{base_name}_labels.npy"
            ),
            labels[:len(ecm_features)]
        )
        print(
            f"Saved: "
            f"ECM {estimator} {ecm_features.shape} | "
            f"LEC {estimator} {lec_features.shape}"
        )
print("\nGeometry feature extraction DONE.")

