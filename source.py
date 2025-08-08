import os
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import resnet50
from collections import OrderedDict
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from warp_adam import WarpAdam
from PIL import Image
import numpy as np

# ===========================
# 1. Device Setup
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_IMAGES = r"D:\MAHESH\library\car.v1i.coco\train"
TRAIN_ANNOTATIONS = r"D:\MAHESH\library\car.v1i.coco\train\_annotations.coco.json"
VAL_IMAGES = r"D:\MAHESH\library\car.v1i.coco\valid"
VAL_ANNOTATIONS = r"D:\MAHESH\library\car.v1i.coco\valid\_annotations.coco.json"

# ===========================
# 2. Spatial Attention Module
# ===========================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

# ===========================
# 3. ResNet-50 with SAM
# ===========================
class ResNet50WithSAM(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50WithSAM, self).__init__()
        base = resnet50(pretrained=pretrained)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.sam = SpatialAttention()
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = self.sam(out2)  # SAM after layer2
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return OrderedDict({
            'layer1': out1,
            'layer2': out2,
            'layer3': out3,
            'layer4': out4
        })

# ===========================
# 4. Faster R-CNN with FPN
# ===========================
def get_fasterrcnn_with_sam(num_classes):
    backbone = ResNet50WithSAM(pretrained=True)
    in_channels_list = [256, 512, 1024, 2048]
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool()
    )
    return FasterRCNN(backbone_with_fpn, num_classes=num_classes)

# ===========================
# 5. COCO Dataset Loader
# ===========================
class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        target = {'image_id': torch.tensor([img_id]), 'boxes': [], 'labels': []}

        coco = COCO(self.annFile)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        if self._transforms:
            img = self._transforms(img)

        return img, target

# ===========================
# 6. Data transforms
# ===========================
transform = T.Compose([T.ToTensor()])
train_dataset = CocoDataset(TRAIN_IMAGES, TRAIN_ANNOTATIONS, transforms=transform)
val_dataset = CocoDataset(VAL_IMAGES, VAL_ANNOTATIONS, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ===========================
# 7. Model + Optimizer
# ===========================
num_classes = 6  # 5 classes + background
model = get_fasterrcnn_with_sam(num_classes).to(device)
optimizer = WarpAdam(model.parameters(), lr=0.00025, weight_decay=1e-4)

# ===========================
# 8. Training Loop
# ===========================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

# ===========================
# 9. Save trained model
# ===========================
torch.save(model.state_dict(), "faster_rcnn_sam_warpadam.pth")
print("Model saved to faster_rcnn_sam_warpadam.pth")

# ===========================
# 10. Inference on One Image
# ===========================
def detect_and_display(model, image_path, class_names, device, score_threshold=0.5):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    output = outputs[0]
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score >= score_threshold:
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[label.item()]
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{class_name}: {score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow("Detection", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===========================
# 11. Run Detection
# ===========================
model_infer = get_fasterrcnn_with_sam(num_classes)
model_infer.load_state_dict(torch.load("faster_rcnn_sam_warpadam.pth", map_location=device))
model_infer.to(device)

class_names = {
    1: "Sleeping",
    2: "Eating",
    3: "Phone Usage",
    4: "Disturbing Others",
    5: "Normal"} # adjust to your labels
TEST_IMAGE = r"D:\MAHESH\library\car.v1i.coco\valid\your_test_image.jpg"
detect_and_display(model_infer, TEST_IMAGE, class_names, device, score_threshold=0.5)
