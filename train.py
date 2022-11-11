import torch
import pandas as pd
import os
import sys
import json
from skimage import io, transform
import torchvision
from tqdm import tqdm
import math
import config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def get_bboxes_from_string(label_string):
    contents = json.loads(label_string.replace("'", ""))
    bboxes_list = []
    if isinstance(contents, list):
        for obj_dict in contents:
            cx = obj_dict['cx']
            cy = obj_dict['cy']
            r = obj_dict['r']
            bbox_dict = {
                'x_min': cx - r,
                'x_max': cx + r,
                'y_min': cy - r,
                'y_max': cy + r
            }
            bboxes_list.append(bbox_dict)
        return bboxes_list
    else:
        return None


class LizaDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder_path, labeling_file_path, width, height):
        self.width = width
        self.height = height
        self.img_folder_path = img_folder_path
        # self.img_names = list(filter(lambda x: 'jpg' in x or 'JPG' in x, os.listdir(img_folder_path)))
        df = pd.read_csv(labeling_file_path)
        df = df[df['count_region'] != 0]
        self.img_names = df['ID_img'].tolist()
        self.labeling_dict = {}
        for row in df.iterrows():
            img_name = row[1]['ID_img']
            label_string = row[1]['region_shape']
            self.labeling_dict[img_name] = get_bboxes_from_string(label_string)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        raw_img = io.imread(os.path.join(self.img_folder_path, self.img_names[idx]))
        w, h = raw_img.shape[1], raw_img.shape[0]
        img = torch.tensor(transform.resize(raw_img, (self.height, self.width)), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        raw_bboxes = self.labeling_dict[self.img_names[idx]]
        bboxes = []
        if raw_bboxes is not None:
            for bbox in raw_bboxes:
                bboxes.append([
                    self.width * bbox['x_min'] / w,
                    self.height * bbox['y_min'] / h,
                    self.width * bbox['x_max'] / w,
                    self.height * bbox['y_max'] / h
                ])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        is_crowd = torch.zeros((len(bboxes),), dtype=torch.int64)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        target = {
            'boxes': bboxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': is_crowd
        }
        return img, target


def get_object_detection_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    print(f'Epoch {epoch + 1} loss: {sum(all_losses) / len(all_losses)}')


if __name__ == "__main__":
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_object_detection_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    dataset = LizaDataset(config.TRAIN_IMG_FOLDER_PATH, config.TRAIN_LABELING_FILE_PATH, config.WIDTH, config.HEIGHT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        train_one_epoch(model, optimizer, loader, device, epoch)
        lr_scheduler.step()
        torch.save(model, f'checkpoints/model_{epoch + 1}.pth')
