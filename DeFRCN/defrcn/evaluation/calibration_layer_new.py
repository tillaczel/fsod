import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity
from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101


# class ContrastiveClassifier:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.device = torch.device(cfg.MODEL.DEVICE)
#
#     def get_scores(self, img, boxes):
#         features = self.extract_roi_features(img, boxes)
#
#         prototypes = torch.cat(list(self.prototypes.values()), dim=0)
#         return cosine_similarity(features.cpu().data.numpy(), prototypes.cpu().data.numpy())
#
#     def extract_roi_features(self, img, boxes):
#         raise NotImplementedError
#
#     def build_image_model(self):
#         raise NotImplementedError
#
#     def build_prototypes(self):
#         raise NotImplementedError
#
# import clip
# from detectron2.data import MetadataCatalog
# from torchvision.transforms import Resize
#
#
# class CLIPContrastiveClassifier(ContrastiveClassifier):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#         self.dataset_dicts = [MetadataCatalog.get(dataset_name) for dataset_name in (cfg.DATASETS.TRAIN+cfg.DATASETS.TEST)]
#
#         self.image_model = self.build_image_model()
#         self.prototypes = self.build_prototypes()
#
#     def build_image_model(self):
#         model, preprocess = clip.load("ViT-B/32", device=self.device)
#         return model
#
#     def build_prototypes(self):
#         classes = set()
#         for dd in self.dataset_dicts:
#             classes.update(dd.thing_classes)
#
#         model, preprocess = clip.load("ViT-B/32", device=self.device)
#         promts = ['a photo of a {} in a scene'.format(c) for c in classes]
#         text = clip.tokenize(promts).to(self.device)
#         with torch.no_grad():
#             text_features = model.encode_text(text)
#
#         prototypes_dict = {}
#         for i in range(text_features.shape[0]):
#             prototypes_dict[i] = text_features[i].unsqueeze(0)
#
#         return prototypes_dict
#
#     def extract_roi_features(self, img, boxes):
#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).to(self.device)
#
#         features = list()
#         resize = Resize((224, 224))
#         for bbox in boxes[0].tensor:
#             cropped_img = img[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].unsqueeze(0)
#             cropped_img = resize(cropped_img)
#             features.append(self.image_model.encode_image(cropped_img))
#         features = torch.cat(features, dim=0)
#         return features
#
#
# class ResnetContrastiveClassifier(ContrastiveClassifier):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#         self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
#         self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
#
#         self.image_model = self.build_image_model()
#         self.prototypes = self.build_prototypes()
#
#     def build_image_model(self):
#         logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
#         if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
#             imagenet_model = resnet101()
#         else:
#             raise NotImplementedError
#         state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
#         imagenet_model.load_state_dict(state_dict)
#         imagenet_model = imagenet_model.to(self.device)
#         imagenet_model.eval()
#         return imagenet_model
#
#     def build_prototypes(self):
#
#         all_features, all_labels = [], []
#         for index in range(len(self.dataloader.dataset)):
#             inputs = [self.dataloader.dataset[index]]
#             assert len(inputs) == 1
#             # load support images and gt-boxes
#             img = cv2.imread(inputs[0]['file_name'])  # BGR
#             img_h, img_w = img.shape[0], img.shape[1]
#             ratio = img_h / inputs[0]['instances'].image_size[0]
#             inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
#             boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]
#
#             # extract roi features
#             features = self.extract_roi_features(img, boxes)
#             all_features.append(features.cpu().data)
#
#             gt_classes = [x['instances'].gt_classes for x in inputs]
#             all_labels.append(gt_classes[0].cpu().data)
#
#         # concat
#         all_features = torch.cat(all_features, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)
#         assert all_features.shape[0] == all_labels.shape[0]
#
#         # calculate prototype
#         features_dict = {}
#         for i, label in enumerate(all_labels):
#             label = int(label)
#             if label not in features_dict:
#                 features_dict[label] = []
#             features_dict[label].append(all_features[i].unsqueeze(0))
#
#         prototypes_dict = {}
#         for label in features_dict:
#             features = torch.cat(features_dict[label], dim=0)
#             prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)
#
#         return prototypes_dict
#
#     def extract_roi_features(self, img, boxes):
#         """
#         :param img:
#         :param boxes:
#         :return:
#         """
#
#         mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
#         std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)
#
#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).to(self.device)
#         images = [(img / 255. - mean) / std]
#         images = ImageList.from_tensors(images, 0)
#         conv_feature = self.image_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW
#
#         box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
#
#         activation_vectors = self.image_model.fc(box_features)
#
#         return activation_vectors


# class PrototypicalCalibrationBlock:
#
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.device = torch.device(cfg.MODEL.DEVICE)
#
#         self.resnet_contrastive_classifier = ResnetContrastiveClassifier(cfg)
#         self.resnet_weight = 0  # Todo
#
#         self.clip_contrastive_classifier = None  # CLIPContrastiveClassifier(cfg)
#         self.clip_weight = 0  # Todo
#
#         self.exclude_cls = self.clsid_filter()
#
#     def execute_calibration(self, inputs, dts):
#
#         img = cv2.imread(inputs[0]['file_name'])
#
#         ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
#         iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
#         if ileft == iright:
#             return dts
#         assert ileft < iright
#         boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]
#
#         if self.resnet_contrastive_classifier is not None:
#             scores = self.resnet_contrastive_classifier.get_scores(img, boxes)
#
#         if self.clip_contrastive_classifier is not None:
#             scores = self.clip_contrastive_classifier.get_scores(img, boxes)
#
#         for i in range(ileft, iright):
#             tmp_class = int(dts[0]['instances'].pred_classes[i])
#             if tmp_class in self.exclude_cls:
#                 continue
#             if self.resnet_contrastive_classifier is not None:
#                 dts[0]['instances'].scores[i] += scores[i-ileft, tmp_class] * self.resnet_weight
#             if self.clip_contrastive_classifier is not None:
#                 dts[0]['instances'].scores[i] += scores[i - ileft, tmp_class] * self.resnet_weight
#             dts[0]['instances'].scores[i] /= (1 + self.resnet_weight + self.clip_weight)
#         return dts
#
#     def clsid_filter(self):
#         dsname = self.cfg.DATASETS.TEST[0]
#         exclude_ids = []
#         if 'test_all' in dsname:
#             if 'coco' in dsname:
#                 exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#                                30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
#                                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
#                                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
#             elif 'voc' in dsname:
#                 exclude_ids = list(range(0, 15))
#             else:
#                 raise NotImplementedError
#         return exclude_ids