import os
import numpy as np
from sklearn.covariance import LedoitWolf, OAS
input_folder = "processed_data_epochs"
output_folder = "geometry_features"
os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))
for file in files:
    if file.endswith("_epochs.npy"):
        base_name = file.replace("_epochs.npy", "")
        print(f"Processing: {base_name}")
        X = np.load(os.path.join(input_folder, f"{base_name}_epochs.npy"))
        labels = np.load(os.path.join(input_folder, f"{base_name}_labels.npy"))
        labels = labels - np.min(labels)
        n_trials = X.shape[0]
        ecm_features = []
        lec_features = []
        for i in range(n_trials):
            trial = X[i]   # shape: (channels, time)
            corr = np.corrcoef(trial)
            if np.isnan(corr).any():
                continue
            lw_cov = LedoitWolf().fit(trial.T).covariance_
            oas_cov = OAS().fit(trial.T).covariance_
            idx = np.tril_indices_from(corr, k=-1)
            ecm = corr[idx]
            lec_matrix = np.log1p(np.abs(corr))
            lec = lec_matrix[idx]
            ecm_features.append(ecm)
            lec_features.append(lec)
        ecm_features = np.array(ecm_features)
        lec_features = np.array(lec_features)
        np.save(
            os.path.join(output_folder, f"{base_name}_ECM.npy"),
            ecm_features
        )
        np.save(
            os.path.join(output_folder, f"{base_name}_LEC.npy"),
            lec_features
        )
        np.save(
            os.path.join(output_folder, f"{base_name}_labels.npy"),
            labels[:len(ecm_features)]  # match skipped trials
        )
print("\nFeature extraction DONE. Saved in 'geometry_features'")