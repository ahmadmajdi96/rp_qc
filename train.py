import shutil
import os
import random
import torch
import torchvision
from PIL import Image

folder = 'Dataset/test'

try:
    shutil.rmtree('Dataset/test')
except:
    print("Testing Dir Does Not Exist")
    
class_names = ['PASS', 'FAIL']
root_dir = 'Dataset'
source_dirs = ['PASS', 'FAIL']

"Dataset/test"

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))
    
    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 700)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)
            
class DatasetProcess(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print("Found the images list")
            return images
        self.images = {}
        self.class_names = ["PASS", "FAIL"]
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
        
        self.image_dirs = image_dirs
        self.transform = transform
    
    def __len__(self):
        return sum(len(self.images[class_name]) for class_name in self.class_names)
    
    def __getitem__(self, index):
        class_name = random.choice[self.class_names]
        index = index % len(self.class_names[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.class_names.index(class_name)
    
train_transform = torchvision.transforms.Compose(
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
)

test_transform = torchvision.transforms.Compose(
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
)


train_dirs= {
    "PASS": "Dataset/PASS",
    "FAIL": "Dataset/FAIL"
}

test_dirs= {
    "PASS": "Dataset/test/PASS",
    "FAIL": "Dataset/test/FAIL"
}

train_dataset = DatasetProcess(train_dirs, train_transform)
test_dataset = DatasetProcess(test_dirs, test_transform)


dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
