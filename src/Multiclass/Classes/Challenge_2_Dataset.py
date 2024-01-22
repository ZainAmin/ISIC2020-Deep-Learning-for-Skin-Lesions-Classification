from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
import numpy as np
import albumentations as A

class Challenge2Dataset(Dataset):
    """
    Custom dataset class for Challenge 2.

    Parameters:
    - csv_file (str): Path to the CSV file containing image information and labels.
    - image_folder (str): Path to the folder containing the images.
    - img_transform (torchvision.transforms.Compose): Image transformations to be applied.
    - augment (bool): If True, apply data augmentation.
    """

    def __init__(self, csv_file, image_folder, img_transform=None, augment=False):
        """
        Initializes the dataset with the given parameters.
        """
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.img_transform = img_transform
        self.augment = augment
        self.labels = self.df["Label"].values.tolist()

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def get_labels(self):
        """
        Returns the list of labels in the dataset.
        """
        return self.labels

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the image and its corresponding label.

        Parameters:
        - idx (int): Index of the sample to be retrieved.

        Returns:
        - sample (dict): Dictionary containing the 'image' and 'label'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = '/' + self.df.iloc[idx]['Image_ID']
        img_path = self.image_folder + img_id[0:4] + img_id
        image = Image.open(img_path)

        label = self.df.iloc[idx]['Label']

        if self.augment:
            image_np = np.array(image)
            aug_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=260),
                    A.RandomResizedCrop(height=256, width=256, p=0.7, scale=(0.4, 1.0), ratio=(0.75, 4/3)),
                    A.Affine(scale=(0.8, 1.2), rotate=(0.0, 90.0), shear=(0.0, 20.0), mode=1, p=0.8),
                    A.Flip(p=0.9),
                    A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.9, 1.1], hue=0.05, p=0.6),
                ]
            )
            augmentations = aug_transform(image=image_np)
            image = Image.fromarray(augmentations["image"])

        if self.img_transform:
            image = self.img_transform(image)

        return {
            'image': image,
            'label': label,
        }