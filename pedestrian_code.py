from __future__ import annotations

__doc__: str = r'''
    >>> Paper Topic: An Efficient Hybrid Resisdual Network for Pedestrian Detection and Segmentation

    >>> Paper Abstract: 
            Pedestrian detection is a vital and important issue which needs to be addressed as it has many applications in different fields such as advanced mechanics, 
            automotive safety, and most importantly surveillance. A large part of the advancement of the previous few years has been driven by the accessibility of 
            testing public datasets and proposing fruitful solutions to the problem. Object detection for the most part requires sliding-window classifiers in custom 
            or anchor-based expectations in present day profound learning draws near. To proceed with the quick pace of advancement, another point of view where 
            identifying objects is inspired as an undeniable level semantic object detection task is presented in this paper and further developed assessment measurements, 
            showing that generally utilized per-window measures are less effective and can neglect to anticipate execution on full pictures are also presented. 
            The proposed hybrid model has the baseline of MobileNet, with the integration of skip connection of ResNet50 with Faster RCNN and focuses all over the 
            picture and extract all the important features that are required for detection. Nonetheless, in contrast to these customary low-level provisions, the 
            proposed model goes for a more elevated level detection. Subsequently, in this paper, pedestrian detection is streamlined as a straight-forward focus and scale 
            expectation task through convolutions. Thusly, the proposed technique is basically basic, it presents competitive precision and great speed and prompts a new 
            attractive pedestrian detector.

'''

import warnings, os, copy, time
warnings.filterwarnings('ignore')
from typing import Optional, ClassVar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metric import auc

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torchvision.models.detection import faster_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



#@: Custom Dataset class 
class PedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, path: 'dir_path', transforms: Optional[torchvision.transforms] = None) -> None:
        self.path = path
        self.transforms = transforms
        self.images: list = sorted([
            image_path for image_path in os.listdir(os.path.join(self.path, 'PNGImages'))
        ])
        self.masks: list = sorted([
            mask_path for mask_path in os.listdir(os.path.join(self.path, 'PedMasks'))
        ])
        
        
    def __len__(self) -> int:
        return len(self.images)
    


    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path =  os.path.join(self.path, 'PNGImages', self.images[index])
        mask_path = os.path.join(self.path, 'PedMasks',  self.masks[index])
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        boxes: list = []
        for i in range(len(obj_ids)):
            pos: int = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.as_tensor(boxes, dtype= torch.float32)
        labels = torch.ones((len(obj_ids), ), dtype= torch.int64)
        image_id = torch.tensor([index])
        return_map: dict[str, torch.Tensor] = {
            'boxes': boxes, 'labels': labels, 'image_id': image_id
        }
        if self.transforms is not None:
            image, return_map = self.transforms(image, return_map)
        
        return image, return_map



#@: Transforms class 
class Compose:
    def __init__(self, transforms: torchvision.transforms) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target



class ToTensor:
    def __call__(self, image: 'image', target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torchvision.transforms.ToTensor()(image)
        return image, target




#@: Utils Functions
def sample_show(image: torch.Tensor, target: dict[str, torch.Tensor], 
                                     show: Optional[bool] = True) -> 'image':
    image: np.ndarray = image.numpy()
    image = image.transpose((1, 2, 0)) * 255
    image = image.astype(np.uint8)

    image: 'image' = Image.fromarray(image)
    boxes = target['boxes'].numpy()
    boxes = np.ceil(boxes)
    
    draw = ImageDraw.Draw(image)
    for box_id in range(boxes.shape[0]):
        box = list(boxes[box_id, :])
        draw.rectangle(box, outline= (255, 0, 255))
    
    if show:
        image.show()
    
    return image


def relu(x: any) -> any: return max(x, 0)

def collate_function(batch: any) -> any: return tuple(zip(*batch))



#@: Evaluation Fnctions
def IOU(dt_box: list| np.ndarray, gt_box: list| np.ndarray) -> float:
    intersection_box = np.array([
        max(dt_box[0], gt_box[0]),
        max(dt_box[1], gt_box[1]),
        min(dt_box[2], gt_box[2]),
        min(dt_box[3], gt_box[3])
    ])

    intersection_area = relu(intersection_box[2] - intersection_box[0]) * relu(intersection_box[3] - intersection_box[1])
    area_dt = (dt_box[2] - dt_box[0]) * (dt_box[3] - dt_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union_area = area_dt + area_gt - intersection_area
    iou = intersection_area/ union_area
    return iou



def evaluate_sample(target_pred: dict[str, torch.Tensor], target_true: dict[str, torch.Tensor], 
                                                          iou_threshold: Optional[float] = 0.5) -> list[dict]:
    gt_boxes = target_true['boxes'].numpy()
    gt_labels = target_true['labels'].numpu()

    dt_boxes = target_pred['boxes'].numpy()
    dt_labels = target_pred['labels'].numpy()
    dt_scores = target_pred['scores'].numpy()

    result: list = []
    for detection_id in range(len(dt_labels)):
        dt_box = dt_boxes[detection_id, :]
        dt_label = dt_labels[detection_id]
        dt_score = dt_scores[detection_id]

        detection_dict: dict = {'score': dt_score}
        max_iou: int = 0
        max_gt_id: int = -1
        for gt_id in range(len(gt_labels)):
            gt_box = gt_boxes[gt_id, :]
            gt_label = gt_labels[gt_id]
            if gt_label != dt_label:
                continue

            if IOU(dt_box, gt_box) > max_iou:
                max_iou = IOU(dt_box, gt_box)
                max_gt_id = gt_id
            
        if max_gt_id >= 0 and max_iou >= iou_threshold:
            detection_dict['TP'] = 1
            gt_labels = np.delete(gt_labels, max_gt_id, axis= 0)
            gt_boxes = np.delete(gt_boxes, max_gt_id, axis= 0)
        else:
            detection_dict['TP'] = 0
        
        result.append(detection_dict)
    
    return result



#@: Net Class
class Net(nn.module):
    def __init__(self) -> None:
        super(Net, self).__init__()

    
    def forward(self) -> 'model':
        num_classes: int = 2
        model = fasterrcnn_resnet_50_fpn(pretrained= True)
        count: int = 0
        for child in model.backbone.children():
            if count == 0:
                for param in child.parameters():
                    param.requires_grad = False
            count += 1
        
        for param in model.roi_heads.box_head.fc6.parameters():
            param.requires_grad = False
        
        in_features = model.roi_heads.box_predictor.cls_score_.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model



#@: Model Adaptor Class
class Model():
    def __init__(self, model: 'net', optimizer: object, 
                                     train_loader: object, 
                                     test_loader: object, 
                                     device: torch.device, 
                                     num_epochs: int) -> None:
        self.net = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
    


    def train_one_epoch(self) -> float:
        self.net.train()
        count: int = 0
        global_loss: float = 0.0
        for images, targets in self.train_loader:
            images = list(images.to(self.device).float() for image in images)
            targets = [
                {key: value.to(self.device) for key, value in t.items()}
                for t in targets
            ]
            dict_loss = self.net(images, targets)
            losses = sum(loss for loss in dict_loss.values())

            self.optimizer.sero_grad()
            losses.backward()
            self.optimizer.step()

            count += 1
            global_loss += float(losses.cpu().detach().numpy())

            if count % 10 == 0:
                print(f'Loss after {count} batches: {round(global_loss/ count, 2)}')
        
        return global_loss




    def evaluate(self) -> tuple[float]:
        results: list = []
        self.net.eval()
        nbr_boxes: int = 0
        with torch.no_grad():
            for batch, (images, targets_true) in enumerate(self.test_loader):
                images = list(image.to(self.device).float() for image in images)
                targets_pred = self.net(images)

                targets_true = [
                    {key: value.cpu().float() for key, value in t.items()}
                    for t in targets_true
                ]
                target_preds = [
                    {key: value.cpu().float() for key, value in t.items()}
                    for t in targets_pred
                ]

                for ii in range(len(targets_true)):
                    target_true = targets_true[ii]
                    target_pred = targets_pred[ii]
                    nbr_boxes += target_true['labels'].shape[0]

                    results = results + evaluate_sample(target_pred, target_true)
        
        results = sorted(results, key= lambda k: k['score'], reverse= True)
        
        acc_TP = np.zeros(len(results))
        acc_FP = np.zeros(len(results))
        recall = np.zeros(len(results))
        precision = np.zeros(len(results))

        if results[o]['TP'] == 1:
            acc_TP[0] = 1
        else:
            acc_FP[0] = 1
        
        for ii in range(1, range(len(results))):
            acc_TP[ii] = results[ii]['TP'] + acc_TP[ii - 1]
            acc_FP[ii] = (1 - results[ii]['TP']) + acc_FP[ii - 1]

            precision = acc_TP[ii] / (acc_TP[ii] + acc_FP[ii])
            recall[ii] = acc_TP[ii] / nbr_boxes

        return auc(recall, precision)




    def train(self) -> None:
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/ {self.num_epochs}')
            start = time.time()
            self.train_one_epoch()
            mAP = self.evaluate()
            end = time.time()

            print(f'mAP after epoch {epoch}: {round(mAP, 3)}')




#@: Driver Code
if __name__.__contains__('__main__'):
    path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\PennFudanPed'

    dataset: object = PedestrianDataset(path, Compose([ToTensor()]))
    indices = torch.randperm(len(dataset)).tolist()

    dataset_train = torch.utils.data.Subset(dataset, indices[: -50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    train_loader: object = torch.utils.data.DataLoader(dataset_train, batch_size= 2, shuffle= True, collate_fn= collate_function)
    test_loader: object = torch.utils.data.DataLoader(dataset_test, batch_size= 2, shuffle= True, collate_fn= collate_function)
    
    if True:
        image, target = dataset[10]
        sample_show(image, target)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Net().to(device)

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    object_detection_model = Model(
        model= model, 
        optimizer= optimizer, 
        train_loader= train_loader, 
        test_loader= test_loader, 
        device= device, 
        num_epochs= 10
    )
    
    object_detection_model.train()


