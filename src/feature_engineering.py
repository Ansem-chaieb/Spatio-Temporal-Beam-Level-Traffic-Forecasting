import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
from typing import List, Union

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import src.config.data_config as dc


def apply_one_hot_encoder(full_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to specified categorical features.
    
    Args:
        full_data (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded features
    """
    features_to_encode = [dc.COLUMN['GNODEB'], dc.COLUMN['CELL'], dc.COLUMN['BEAM']]

    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(full_data[features_to_encode])
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(features_to_encode),
        index=full_data.index
    )

    return pd.concat([full_data.drop(columns=features_to_encode), encoded_df], axis=1)

def apply_pca(full_data: pd.DataFrame, pca_features: List[str], n_components: int = 5) -> pd.DataFrame:
    """
    Apply PCA to specified features.
    
    Args:
        full_data (pd.DataFrame): Input DataFrame
        pca_features (List[str]): List of feature names to apply PCA on
        n_components (int): Number of principal components to keep
    
    Returns:
        pd.DataFrame: DataFrame with PCA components added
    """
    X = full_data[pca_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    principal_df = pd.DataFrame(data=principal_components, columns=pca_columns, index=full_data.index)

    full_data_pca = pd.concat([full_data, principal_df], axis=1)
    return full_data_pca[dc.FINAL_PCA_FEATURES]

def apply_groupby(full_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply groupby operations to calculate mean and std of traffic per gNodeB.
    
    Args:
        full_data (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with added mean and std traffic features
    """
    groupby_cols = [dc.COLUMN['GNODEB'], dc.COLUMN['CELL'], dc.COLUMN['BEAM']]
    
    full_data['mean_traffic_per_gnodeb'] = full_data.groupby(groupby_cols)[dc.TARGET_COLUMN ].transform('mean')
    full_data['std_traffic_per_gnodeb'] = full_data.groupby(groupby_cols)[dc.TARGET_COLUMN].transform('std')
    
    return full_data

def preprocess_data(full_data: pd.DataFrame, pca_features: List[str]) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the input data.
    
    Args:
        full_data (pd.DataFrame): Input DataFrame
        pca_features (List[str]): List of feature names to apply PCA on
    
    Returns:
        pd.DataFrame: Fully preprocessed DataFrame
    """
    full_data = apply_one_hot_encoder(full_data)
    full_data = apply_groupby(full_data)
    full_data = apply_pca(full_data, pca_features)
    
    return full_data