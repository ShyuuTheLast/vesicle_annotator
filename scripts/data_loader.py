import h5py
import torch
import torch.utils.data as data
import numpy as np

class CustomDataset(data.Dataset):
    def __init__(self, images, masks, labels=None, ids=None, transform=None):
        self.images = images
        self.masks = masks
        # Convert labels from 1-based to 0-based indexing if labels are provided
        self.labels = labels - 1 if labels is not None else np.full(len(images), -1)
        self.ids = ids if ids is not None else np.full(len(images), -1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        index = self.ids[idx]

        image = image.transpose(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
            image = (image - image.mean()) / image.std()

        return image, mask, label, index

def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        return {key: np.array(f[key]) for key in f.keys()}

def load_ids(bounding_box_file):
    with h5py.File(bounding_box_file, 'r') as bb_file:
        return bb_file["main"][:, 0]  # Extract and return the first column (segmentation IDs)

def create_data_loader(image_file, mask_file, label_file=None, bounding_box_file=None, batch_size=128, shuffle=True):
    images = load_hdf5(image_file)['main']
    masks = load_hdf5(mask_file)['main']
    labels = load_hdf5(label_file)['main'] if label_file is not None else None
    ids = load_ids(bounding_box_file) if bounding_box_file is not None else None

    dataset = CustomDataset(images, masks, labels, ids, transform=None)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
