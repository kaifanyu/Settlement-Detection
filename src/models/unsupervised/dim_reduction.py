import sys
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule


def preprocess_for_dim_reduction(
    esd_datamodule: ESDDataModule,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data for the dimensionality reduction

    Input:
        esd_datamodule: ESDDataModule
            datamodule to load the data from

    Output:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

        y_flat: np.ndarray
            Flattened ground truth of shape (sample, 1)
    """
    # --- start here ---
    # create two lists to store the samples for the data (X) and gt (y)
    X_samples = []
    y_samples = []
    
    # for each X, y in the train dataloader
    for X,y in esd_datamodule.train_dataloader():
        # append X to its list
        X_samples.append(X)
        # append y to its list
        y_samples.append(y)

    # concatenate both lists
    X_samples = np.concatenate(X_samples)
    y_samples = np.concatenate(y_samples)
    
    # the X list will now have shape (sample, time, band, width, height)
    X_sample,X_band,X_width,X_height = X_samples.shape
    
    # the y list will now have shape (sample, 1, 1, width, height)
    y_sample,_,_ = y_samples.shape
    
    # reshape the X list to the new shape (sample,time*band*width*height)
    X_flat = np.reshape(X_samples,newshape=((X_sample,X_band*X_width*X_height)))
    
    # reshape the y list to the new shape (sample, 1)
    y_flat = np.reshape(y_samples,newshape=(y_sample,1))

    # return the reshaped X and y
    return X_flat,y_flat


def perform_PCA(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    Perform PCA on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        pca: PCA
            PCA object
    """
    pca = PCA(n_components=n_components)

    return pca.fit_transform(X_flat),pca

def perform_TSNE(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, TSNE]:
    """
    Perform TSNE on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        tsne: TSNE
            TSNE object
    """
    tsne = TSNE(n_components=n_components)
    
    return tsne.fit_transform(X_flat),tsne


def perform_UMAP(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, UMAP]:
    """
    Perform UMAP on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        umap: UMAP
            UMAP object
    """
    umap = UMAP(n_components=n_components)
    
    return umap.fit_transform(X_flat),umap