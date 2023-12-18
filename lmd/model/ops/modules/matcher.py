# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from einops import rearrange, reduce, repeat

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher_Seq(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_feats_weight = [1],
                 num_feat = 4,
                 shared_voca = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_feats_weight = cost_feats_weight
        self.num_feat = num_feat
        self.shared_voca = shared_voca

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            # not Seq: outputs["pred_logits"].shape = [bs, 300, 91]
            # Seq: outputs["pred_logits"].shape = [num_feat, bs, 300, voca_size]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Todo 4.1.1 --------- matcher
        with torch.no_grad():
            if len(outputs["pred_logits"].shape) == 3: # seq output, numfeat>1
                outputs["pred_logits"] = rearrange(outputs["pred_logits"], 'b q v -> 1 b q v')

            # Calculate bbox loss
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Calculate classification loss
            # assert len(targets[0]["labels"]) == len(self.cost_feats_weight), \
            #     f'++++++++++++++++++length of cost_feats_weight must be equal to length of targets["labels"] ++++++++++++'
            # assert len(self.cost_feats_weight) == len(outputs["pred_boxes_logit"]) + len(outputs["pred_logits"]), \
            #     f'++++++++++++++++++length of cost_feats_weight must be equal to len(outputs["pred_boxes_logit"]) + len(outputs["pred_logits"])'

            cost_classes = 0
            for feat_id in range(len(self.cost_feats_weight)):
                # if self.shared_voca:
                pred_logit = outputs["pred_logits"][feat_id]
                # else:
                #     pred_logit = outputs["pred_boxes_logit"][feat_id] if feat_id<4 else outputs["pred_logits"][feat_id-4]

                bs, num_queries = pred_logit.shape[:2]
                # We flatten to compute the cost matrices in a batch
                out_prob = pred_logit.flatten(0, 1).sigmoid()

                # Also concat the target labels and boxes
                tgt_ids = torch.cat([v["labels"][feat_id] for v in targets])

                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                cost_classes += cost_class * self.cost_feats_weight[feat_id]

            # Final cost matrix
            # print(f'+++++++++++size cost_bbox.shape: {cost_bbox.shape}++++++++++++++++++++++')
            # print(f'+++++++++++size cost_giou.shape: {cost_giou.shape}++++++++++++++++++++++')
            # print(f'+++++++++++size cost_class.shape: {cost_class.shape}++++++++++++++++++++++')
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_classes + self.cost_giou * cost_giou
            # print(f'+++++++++++before size (bs, num_queries: {(bs, num_queries)}++++++++++++++++++++++')
            C = C.view(bs, num_queries, -1)
            # print(f'+++++++++++before size C.shape: {C.shape}++++++++++++++++++++++')
            C = C.cpu()
            # print(f'+++++++++++after size c.shape: {C.shape}++++++++++++++++++++++')

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.seq_output:
        matcher = HungarianMatcher_Seq(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                       cost_giou=args.set_cost_giou, cost_feats_weight=args.cost_feats_weight,
                                       shared_voca=args.shared_voca)
    else:
        matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                   cost_giou=args.set_cost_giou)

    return matcher