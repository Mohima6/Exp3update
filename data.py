import os
import mne
import numpy as np
main_folder = "files"
processed_data_folder = "processed_data_epochs"
os.makedirs(processed_data_folder, exist_ok=True)
for subject_folder in sorted(os.listdir(main_folder)):
    subject_path = os.path.join(main_folder, subject_folder)
    if os.path.isdir(subject_path):
        for file in sorted(os.listdir(subject_path)):
            if file.endswith(".edf"):
                file_path = os.path.join(subject_path, file)
                print(f"Processing: {file_path}")
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                raw.pick(raw.ch_names[:32])
                bad_channels = ['Cz', 'Iz']
                raw.drop_channels([ch for ch in bad_channels if ch in raw.ch_names])
                raw.filter(8., 30., verbose=False)
                events, event_dict = mne.events_from_annotations(raw)
                event_id = {}
                for key in event_dict:
                    if "T1" in key:
                        event_id['T1'] = event_dict[key]
                    if "T2" in key:
                        event_id['T2'] = event_dict[key]
                if len(event_id) == 0:
                    print("No valid events found, skipping...")
                    continue
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=0.0,
                    tmax=2.0,
                    baseline=None,
                    preload=True,
                    verbose=False
                )
                data = epochs.get_data()
                labels = epochs.events[:, -1]
                base_name = f"{subject_folder}_{file.replace('.edf','')}"
                np.save(
                    os.path.join(processed_data_folder, f"{base_name}_epochs.npy"),
                    data
                )
                np.save(
                    os.path.join(processed_data_folder, f"{base_name}_labels.npy"),
                    labels
                )
                del raw, epochs, data