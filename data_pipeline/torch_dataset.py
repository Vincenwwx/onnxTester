from torchvision import transforms, datasets
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform


def torch_dataloader(data_dir, input_size, batch_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4) for x in ['train', 'val']}


class Torch_Dataset(Dataset):
    """ Characterizes a dataset for PyTorch """
    def __init__(self, image_path_list, labels_list, transform=None):
        """ Initialization """
        self.labels_list = labels_list
        self.image_path_list = image_path_list

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.image_path_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        """ Generates one sample of data """
        image_path = self.image_path_list[index]
        # Load data and get label
        image = io.imread(image_path[index])
        y = self.labels_list[index]

        return X, y
