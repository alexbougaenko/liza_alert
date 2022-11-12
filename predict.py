import torch
import os
import config
from skimage import io
import json
import pandas as pd
from tqdm import tqdm
import pickle
import torchvision

MODEL_PATH = 'checkpoints/model_10.pth'
DEBUG = True


def apply_nms(orig_prediction, iou_thresh=0.3):

    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(MODEL_PATH)
    model.eval()
    model.to(device)

    debug_list = []
    answers_list = []
    for img_name in tqdm(os.listdir(config.TEST_DATA_PATH)):
        if 'jpg' in img_name or 'JPG' in img_name:
            img = io.imread(os.path.join(config.TEST_DATA_PATH, img_name))
            img = torch.tensor(img, dtype=torch.float32)
            img = img / 255.0
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.to(device)
            with torch.no_grad():
                prediction = model(img)[0]

            # prediction = apply_nms(prediction)

            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()

            if DEBUG:
                debug_list.append((img_name, boxes, scores, labels))

            results_list = []
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.8 and label == 1:
                    x_min, y_min, x_max, y_max = box
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    r = max(x_max - x_min, y_max - y_min) // 2
                    d = {
                        'cx': cx,
                        'cy': cy,
                        'r': r
                    }
                    results_list.append(f"'{json.dumps(d)}'")
            if len(results_list) == 0:
                result_string = '0'
            else:
                result_string = '[' + ', '.join(results_list) + ']'
            answers_list.append([img_name, result_string])

            torch.cuda.empty_cache()

    df = pd.DataFrame(answers_list, columns=['ID_img', 'region_shape'])
    df.to_csv('answers.csv', index=False)
    if DEBUG:
        with open('debug.pkl', 'wb') as f:
            pickle.dump(debug_list, f)
