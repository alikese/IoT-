import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class NpyFlowDataset(Dataset):
    """
    PyTorch Dataset for CICIDS 2018 flow data.
    Features are float32, labels are long for CrossEntropyLoss.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)   # 特征 float32
        self.y = torch.tensor(y, dtype=torch.long)      # 标签 long

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data(args):
    """
    This function loads the dataset based on the user's choice and returns the training and testing data.
    It supports two datasets: 'cicids-2018' and 'unsw-nb15'.
    """
    
    if args.dataset == 'cicids-2018':
        # List of attack classes in CICIDS-2018 dataset, including 'BENIGN' as the normal class
        attack_classes = ['Benign','Bot','DDoS','DoS GoldenEye','DoS Hulk','DoS SlowHTTPTest',
                            'DoS Slowloris','FTP-BruteForce','Heartbleed','Infiltration','PortScan',
                            'SSH-BruteForce','Sql Injection','Web Attack - XSS','Web-BruteForce']
        
        # The number of classes selected for training (based on user's selection)
        num_classes = len(args.selected_class)

        # Load the CICIDS-2018 data, splitting into training and testing sets
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
            get_data_CICIDS_2018(args.attack_max_samples,args.selected_class, args.test_split_size)

    # Get the number of features from the training data (columns)
    feature_num = X_train.shape[1]

    # Ensure that the feature_min and feature_max have the same number of dimensions as the feature data
    assert feature_num == len(feature_min)
    assert feature_num == len(feature_max)

    # Return the processed data and metadata
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list, num_classes, feature_num, attack_classes


def get_data_CICIDS_2018(iscx_attack_max_samples,iscx_selected_class, test_split_size):
    """
    This function loads and processes the CICIDS-2018 dataset. It selects data based on the user's class selection
    and splits it into training and testing sets.
    """
    
    # Load the features and labels for the CICIDS-2018 dataset
    CUSTOM_LABEL_COL = 'Label'
    df = pd.read_csv(os.path.join('./CIC17_18/CIC17_18/CICIDS_2017_all_days_binary.csv'))
    data = df.drop(columns=[CUSTOM_LABEL_COL]).to_numpy()
    label = df[CUSTOM_LABEL_COL].astype(str).to_numpy()
    iscx_selected_class = [str(c) for c in iscx_selected_class]

    selected_data = []
    selected_label = []

    # Loop over the selected attack classes and gather the corresponding data
    # for idx, attack_class in enumerate(iscx_selected_class):
    #     # Select up to 'iscx_attack_max_samples' samples from each class
    #     selected_data.append(data[label == attack_class][:iscx_attack_max_samples])
    #
    #     # Create labels corresponding to the class
    #     length = data[label == attack_class][:iscx_attack_max_samples].shape[0]
    #     selected_label.append(np.ones(length) * idx)

    for idx, attack_class in enumerate(iscx_selected_class):
        class_mask = label == attack_class
        class_data = data[class_mask][:iscx_attack_max_samples]
        if class_data.shape[0] == 0:
            print(f"Warning: no samples found for class {attack_class}, skipping")
            continue
        selected_data.append(class_data)
        selected_label.append(np.ones(class_data.shape[0]) * idx)
    
    # Stack the selected data and labels into one dataset
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)

    # Calculate the minimum and maximum values for feature scaling
    feature_max = np.max(selected_data, axis=0)
    feature_min = np.min(selected_data, axis=0)

    # If any feature has the same min and max value (i.e., constant), fix the values to avoid division by zero
    if 0 in (feature_max - feature_min):
        idx0 = (feature_max - feature_min) == 0
        feature_min[idx0] = 0
        feature_max[idx0] = 1

    #selected_data = (selected_data - feature_min) / (feature_max - feature_min + 1e-12)
    # Split the selected data into training and testing sets using the specified test split size
    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2024)

    # Prepare lists to store data from unknown attack classes (those not selected for training)
    unknown_data_list = []
    unknown_label_list = []

    all_classes = np.unique(label)
    for attack_class in all_classes:
        if attack_class in iscx_selected_class:
            continue
        class_mask = label == attack_class
        class_data = data[class_mask][:iscx_attack_max_samples]
        if class_data.shape[0] == 0:
            continue
        unknown_data_list.append(class_data)
        unknown_label_list.append(label[class_mask][:iscx_attack_max_samples])

    # Return the processed data and metadata
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list

