import torch
import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetectorFcos(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 fcos_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorFcos, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if fcos_head is not None:
            self.fcos_head = builder.build_head(fcos_head)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_fcos(self):
        return hasattr(self, 'fcos_head') and self.fcos_head is not None

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorFcos, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

        if self.with_fcos:
            self.fcos_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # print('in single_stage')
        # import pdb
        # pdb.set_trace()

        x = self.extract_feat(img)

        losses = dict()

        if self.with_fcos:
            fcos_outs = self.fcos_head(x)
            fcos_loss_inputs = fcos_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg.fcos)
            fcos_losses = self.fcos_head.loss(
                *fcos_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(fcos_losses)

            bboxes_pred_inputs = fcos_outs + (img_metas, self.test_cfg.fcos)
            fcos_bboxes_pred = self.fcos_head.get_bboxes(*bboxes_pred_inputs)

            fcos_p_bboxes = [0] * len(fcos_bboxes_pred)
            for i in range(len(fcos_bboxes_pred)):
                fcos_p_bboxes_p3 = fcos_bboxes_pred[i][2][0]
                index1 = torch.gt(fcos_p_bboxes_p3[:,4], 0.01)
                index2 = torch.nonzero(index1)
                fcos_p3_proposal_tensor = torch.squeeze(fcos_p_bboxes_p3[index2])
                if len(fcos_p3_proposal_tensor.shape) >= 2:
                    fcos_p_bboxes[i] = fcos_p3_proposal_tensor
                else:
                    fcos_p_bboxes[i] = fcos_p3_proposal_tensor.unsqueeze(0)




        if self.with_bbox:

            bbox_outs = self.bbox_head(x)
            bbox_loss_inputs = bbox_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg.retinanet)
            # TODO: make if flexible to add the bbox_head
            bbox_losses = self.bbox_head.loss(
                *bbox_loss_inputs, fcos_p_bboxes, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(bbox_losses)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
