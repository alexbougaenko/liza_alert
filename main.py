import torch
import pandas as pd
import os
import glob
import json
from skimage import io


def get_bboxes_from_string(label_string):
    contents = json.loads(label_string.replace("'", ""))
    bboxes_list = []
    if isinstance(contents, list):
        for obj_dict in contents:
            cx = obj_dict['cx']
            cy = obj_dict['cy']
            r = obj_dict['r']
            x_0, x_1 = cx - r, cx + r
            y_0, y_1 = cy - r, cy + r
            bboxes_list.append((x_0, x_1, y_0, y_1))
        return bboxes_list
    else:
        return None


class LizaDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder_path, labeling_file_path):
        self.img_paths = glob.glob(os.path.join(img_folder_path, "*.JPG"))
        df = pd.read_csv(labeling_file_path)
        self.labeling_dict = {}
        for row in df.iterrows():
            img_name = row[1]['ID_img']
            label_string = row[1]['region_shape']
            self.labeling_dict[img_name] = get_bboxes_from_string(label_string)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample = {
            'image': io.imread(self.img_paths[idx]),
            'bboxes': self.labeling_dict[os.path.basename(self.img_paths[idx])]
        }
        return sample


if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")
    LizaDataset("data/train", "data/train.csv")
    print()
