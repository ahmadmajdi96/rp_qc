import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.annotation_files = [f.replace('.jpg', '.xml') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        annotation_path = os.path.join(self.root_dir, self.annotation_files[idx])

        image = Image.open(img_path).convert("RGB")
        annotation = self.parse_xml(annotation_path)

        if self.transforms:
            image, annotation = self.transforms(image, annotation)

        return image, annotation

    def parse_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)  # You might want to convert class names to class indices

        return {'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels)}


def custom_transforms(image, target):
    image = ToTensor()(image)
    return image, target

dataset = CustomDataset(root_dir='path/to/your/dataset', transforms=custom_transforms)

image, annotation = dataset[0]
print(image.shape, annotation)

data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

backbone = torchvision.models.resnet50(pretrained=True)
backbone.out_channels = 256

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=91,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()

torch.save(model.state_dict(), 'object_detection_model.pth')
