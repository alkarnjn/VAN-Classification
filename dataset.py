import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomValidationDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        self.classes = sorted(os.listdir(data_folder), key=int)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for class_name in self.classes:
            class_folder = os.path.join(self.data_folder, class_name)
            class_idx = self.class_to_idx[class_name]
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                images.append((image_path, class_idx, class_name))
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
        
        transform = transforms.ToTensor()

        # Convert the PIL image to a PyTorch tensor
        image = transform(image)
        if self.transform:
            image = self.transform(image)
        # print(image_path, label, class_name)
        return image, label, image_path
