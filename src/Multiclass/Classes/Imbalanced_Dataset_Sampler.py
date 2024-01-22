from typing import Callable
import torch
import pandas as pd


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Sampler for imbalanced datasets, ensuring balanced class representation during training.

    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset to sample from.
    - indices (list): List of indices to sample from. Default is None.
    - num_samples (int): Number of samples to draw. Default is None (uses the length of indices).
    - callback_get_label (Callable): Callback function to get labels from the dataset. Default is None.
    - replacement (bool): If True, samples with replacement. Default is True.
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None,
                 replacement=True):
        self.indices = list(range(len(dataset))) if indices is None else indices

        # Define custom callback
        self.callback_get_label = callback_get_label

        self.num_samples = len(self.indices) if num_samples is None else num_samples

        self.replacement = replacement

        # Distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in
                torch.multinomial(self.weights, self.num_samples, replacement=self.replacement))

    def __len__(self):
        return self.num_samples