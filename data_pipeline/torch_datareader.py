import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class Torch_DataReader(Dataset):
    """ Characterizes a dataset for PyTorch """
    def __init__(self, image_path_list, labels_list, transform=None, num_classes=91):
        """ Initialization """
        assert len(image_path_list) == len(labels_list), \
            "Length of image list and label list do not coordinate!"
        self.labels_list = labels_list
        self.image_path_list = image_path_list
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.image_path_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        """ Generates one sample of data """
        image_path = str(self.image_path_list[index])
        # read image into tensor
        try:
            image = Image.open(image_path).convert('RGB')   # some pictures in COCO are B&W
        except:
            print("Error happened when load {} with index {}".format(image_path, index))
            raise
        """ Apply transformers """
        if self.transform:
            image = self.transform(image)

        # load labels
        # labels = torch.zeros(self.num_classes)
        # for label in self.labels_list[index]:
        #     labels += torch.nn.functional.one_hot(torch.tensor(label), self.num_classes)
        labels = torch.sum(torch.nn.functional.one_hot(torch.tensor(self.labels_list[index]), self.num_classes),
                           dim=0)

        return image, labels


def torch_load_and_preprocess_single_img(image_path, size):

    my_transforms = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print("Error happened when load {}".format(image_path))
        raise

    return my_transforms(image).unsqueeze(0)