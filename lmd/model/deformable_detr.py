# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from einops import rearrange, repeat, reduce
import copy

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from model.ops.modules.matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from data_module.cityflow_data import cityflow_build_voca, cityflow_feats
from util.box_ops import box_xyxy_to_cxcywh



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_voca_ffn(input_dim, hidden_dim, num_classes, n_coor_bins, num_layer, inter_output=-2, num_feat=5, prior_prob = 0.01):
    assert num_feat > 4, f'num_feat should > 4 since first 4 feats are b_i'
    bbox_ffn = [copy.deepcopy(MLP(input_dim, hidden_dim, n_coor_bins, num_layer, inter_output, prior_prob )) for i in range(4)]
    feat_ffn = [copy.deepcopy(MLP(input_dim, hidden_dim, num_classes, num_layer, inter_output, prior_prob )) for i in
                range(num_feat - 4)]
    voca_ffn = nn.ModuleList(bbox_ffn + feat_ffn)
    return voca_ffn


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, num_group_norm,
                 aux_loss=True, with_box_refine=False, two_stage=False, num_feat=5, seq_output=False, n_coor_bins=1000,
                 shared_voca = False, xywh=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.xywh = xywh
        self.num_classes = num_classes
        self.n_coor_bins = n_coor_bins
        self.n_feat = num_classes - n_coor_bins
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.shared_voca = shared_voca
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if seq_output:
            prior_prob = 0.01
            if self.shared_voca:
                self.voca_ffn = _get_voca_ffn(hidden_dim, hidden_dim, num_classes, num_classes, 3, inter_output=-2,
                                          num_feat=num_feat, prior_prob = prior_prob)
            else:
                self.voca_ffn = _get_voca_ffn(hidden_dim, hidden_dim, self.n_feat, n_coor_bins, 3, inter_output=-2,
                                          num_feat=num_feat, prior_prob = prior_prob)
            if aux_loss:
                # Get vocas.shape = [num_pred, num_feat]; each element is  MLP(hidden_dim, hidden_dim, num_classes, 3)
                self.voca_ffn = nn.ModuleList([self.voca_ffn for _ in range(num_pred)])


        self.num_feature_levels = num_feature_levels
        self.num_feat = num_feat
        self.seq_output = seq_output
        if not two_stage:
            # why hidde_dim*2 => first half is query embed, 2nd half is used for predicting emb coordinate
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(num_group_norm, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_group_norm, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if not self.seq_output:
            self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, add one more class_embed and bbox_embed for region proposal generation in encoding embed
        if not self.seq_output:
            if with_box_refine:
                # if with_box_refine, self.class_embed layers must be different
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
                nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
                # hack implementation for iterative bounding box refinement
                # If turn on two_stage but no use with_box_refine, problem occur here since encoder emb feed to
                # decoder.bbox_embed to get the b_i
                self.transformer.decoder.bbox_embed = self.bbox_embed
            else:
                nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
                self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # fixme: check if I need this function
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # backbone here is ResNet + positional_encoding
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        #  Loop through different scale level
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            # run linear project (nn.Conv2d + nn.GroupNorm)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        #  If num_feature_levels > num output from resnet => why the batch may transform to multi-scale ?
        # Loop through the extend level and produce feat
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            # loop from len(output_ResNet to num_feature_levels
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # hs, init_reference_out, inter_references_out, None, None
        # return output from all layers of transformer decoder
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        if self.seq_output:
            outputs_classes = []
            outputs_coord_ces = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                # may used for splitting the decoding ouput
                # hs = rearrange(hs, 'l b q (feat dim) -> feat l b q dim', feat=6)

                #  hs.shape = l b q dim = 4 5 300 256
                # Todo 2.1 ----------final ffn for seq
                outputs = [self.voca_ffn[lvl][feat](hs[lvl]) for feat in range(self.num_feat)]
                # outputs_token_class.shape is: f l b q v -> 6 4 5 300 2100
                outputs_token_class = [output[0] for output in outputs]
                # outputs_token_embed = [output[1] for output in outputs]  # used for generating token embedding later

                outputs_coord_ce = torch.stack(outputs_token_class[:4])
                outputs_coord = box_ops.get_normal_coord(outputs_coord_ce, n_coor_bins=self.n_coor_bins)
                # outputs_class = torch.stack(outputs_token_class[4:])  # [f b q v] f is num_feat
                # [f b q v] f is num_feat #get 4 b_i class
                if not self.shared_voca:
                    for idx_f in range(len(outputs_token_class)):
                        if idx_f<4:
                            outputs_token_class[idx_f] = F.pad(input=outputs_token_class[idx_f],
                                                               pad=(0, self.n_feat), mode='constant', value=0)
                        else:
                            outputs_token_class[idx_f] = F.pad(input=outputs_token_class[idx_f], pad=(self.n_coor_bins, 0), mode='constant', value=0)
                outputs_class = torch.stack(outputs_token_class)

                outputs_classes.append(outputs_class)
                outputs_coord_ces.append(outputs_coord_ce)
                outputs_coords.append(outputs_coord)

                # outputs_token_classes.append(outputs_token_class)

            outputs_class = torch.stack(outputs_classes)  # [l f b q v] f is num_feat
            # outputs_class = outputs_classes  # [l f b q v] f is num_feat
            outputs_coord_ce = torch.stack(outputs_coord_ces)  # [l b q 4]

            # Convert to xywh to calculate hugarian loss in matching
            if not self.xywh:
                outputs_coords = [box_xyxy_to_cxcywh(ele) for ele in outputs_coords]
            outputs_coord = torch.stack(outputs_coords)  # [l b q 4]

            out = {'pred_logits': outputs_class[-1], 'pred_boxes_logit': outputs_coord_ce[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_ce, outputs_coord, self.shared_voca)

            if self.two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            return out

        else:
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord = outputs_coord)

            if self.two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord_ce=None, outputs_coord = None, shared_voca = False):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if len(outputs_class.shape) == 4: # not seq
        # if not shared_voca:
        #     aux_loss = [{'pred_logits': a, 'pred_boxes_logit': b, 'pred_boxes': c}
        #                 for a, b, c in zip(outputs_class[:-1], outputs_coord_ce[:-1], outputs_coord[:-1])]
        # else:
        aux_loss = [{'pred_logits': a,  'pred_boxes': b}
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return aux_loss


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, cost_feats_weight=[1],
                 shared_voca = False, n_coor_bins=1000, focal_gamma = 2
                 ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.cost_feats_weight = cost_feats_weight
        self.shared_voca = shared_voca
        self.focal_gamma = focal_gamma
        self.empty_weight = None

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # Todo 4.1.2 ----------------Loss label
        if len(outputs["pred_logits"].shape) == 3:  # seq output, num_feat>1
            outputs["pred_logits"] = rearrange(outputs["pred_logits"], 'b q v -> 1 b q v')

        # assert len(targets[0]["labels"]) == len(self.cost_feats_weight), \
        #     f'++++++++++++++++++length of cost_feats_weight must be equal to length of targets["labels"] ++++++++++++'
        # assert len(self.cost_feats_weight) == len(outputs["pred_boxes_logit"]) + len(outputs["pred_logits"]), \
        #     f'++++++++++++++++++length of cost_feats_weight must be equal to len(outputs["pred_boxes_logit"]) + len(outputs["pred_logits"])'

        # print (f"+++++++outputs['pred_logits'].shape: {outputs['pred_logits'].shape} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++outputs['pred_boxes'].shape: {outputs['pred_boxes'].shape} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++outputs['pred_logits']: {outputs['pred_logits']} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++outputs['pred_boxes']: {outputs['pred_boxes']} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++targets: {targets} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++indices: {indices} +++++++++++++++++++++++++++++++++++")
        # print (f"+++++++num_boxes: {num_boxes} +++++++++++++++++++++++++++++++++++")

        loss_ces = 0
        for feat_id in range(len(self.cost_feats_weight)):
            # if self.shared_voca:
            pred_logit = outputs["pred_logits"][feat_id]
            # else:
            #     pred_logit = outputs["pred_boxes_logit"][feat_id] if feat_id<4 else outputs["pred_logits"][feat_id-4]
            # src_logits = outputs['pred_logits'][feat_id]

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][feat_id][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(pred_logit.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=pred_logit.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([pred_logit.shape[0], pred_logit.shape[1], pred_logit.shape[2] + 1],
                                                dtype=pred_logit.dtype, layout=pred_logit.layout,
                                                device=pred_logit.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(pred_logit, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                            gamma=self.focal_gamma) * pred_logit.shape[1]
            loss_ces += loss_ce * self.cost_feats_weight[feat_id]
        # hard code loss_ce>600
        loss_ces = loss_ces /600
        losses = {'loss_ce': loss_ces}

        if log:
            losses['class_error'] = 100 - accuracy(pred_logit[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # if len(outputs["pred_logits"].shape) == 3:  # seq output, numfeat>1
        #     outputs["pred_logits"] = rearrange(outputs["pred_logits"], 'b q v -> 1 b q v')

        pred_logits = outputs['pred_logits'][0]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"][0]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                     {'pred_logits': , 'pred_boxes': , 'aux_outputs', 'enc_outputs'}
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      example of keys = ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # print(f'++++++++++++++++targets: {targets}++++++++++++++++++++++++')
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # losses is ['labels', 'boxes', 'cardinality']
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # Todo 4.1 --------- new SetCriterion - loss function - auxiliary losses
        # The problem may occur here since the tuple element now is [f b q v] not [b q v]

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, seq_output = False, cost_feats_weight=[1], shared_voca = False, n_coor_bins=1000, n_f1=91,
                 dataset_file= 'coco_data', xywh=True):
        super().__init__()
        self.seq_output = seq_output
        self.cost_feats_weight = cost_feats_weight
        self.shared_voca = shared_voca
        self.n_coor_bins = n_coor_bins
        self.n_f1=n_f1
        self.dataset_file = dataset_file
        self.xywh = xywh
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data_module augmentation)
                          For visualization, this should be the image size after data_module augment, but before padding
        """
        # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_bbox = outputs['pred_boxes']

        if self.seq_output:
            assert len(outputs["pred_logits"][0]) == len(target_sizes)
        else:
            assert len(outputs["pred_logits"]) == len(target_sizes)

        assert target_sizes.shape[1] == 2

        if len(outputs["pred_logits"].shape) == 3:  # seq output, num_feat>1
            outputs["pred_logits"] = rearrange(outputs["pred_logits"], 'b q v -> 1 b q v')

        prob_views = []
        # Concatenate b_i and other f_j (j>1)
        # for feat_id in range(len(self.cost_feats_weight)):
        #     # Todo 4.1.3 ----------postprocessor -> use b_i for selecting the top 100 query ?
        #     if self.shared_voca:
        #         pred_logit = outputs["pred_logits"][feat_id]
        #     else:
        #         pred_logit = outputs["pred_boxes_logit"][feat_id] if feat_id<4 else outputs["pred_logits"][feat_id-4]
        #     # Todo 5a.1 -------------why use sigmoid here ? => need to check how compute the ce loss
        #     prob = pred_logit.sigmoid()
        #     prob = prob[:, :, self.n_coor_bins:self.n_coor_bins + self.n_f1].clone()
        #     # 'b q v -> b q*v'.
        #     #  b q*v = b [v_1 v_2 .. v_q]
        #     prob_view = prob.view(pred_logit.shape[0], -1)
        #     prob_views.append(prob_view*self.cost_feats_weight[feat_id])
        # prob_views = rearrange(prob_views, 'f b qv -> f b qv')
        # prob_views = reduce(prob_views, 'f b qv -> b qv', 'sum')

        # Only consider f_1
            # Todo 4.1.3 ----------postprocessor -> use b_i for selecting the top 100 query ?
        # if self.shared_voca:
        pred_logit = outputs["pred_logits"][4]
        # else:
        #     pred_logit = outputs["pred_logits"][0]
        # Todo 5a.1 -------------why use sigmoid here ? => need to check how compute the ce loss
        prob = pred_logit.sigmoid()
        prob = prob[:, :, self.n_coor_bins:self.n_coor_bins + self.n_f1].clone()
        # 'b q v -> b q*v'.
        #  b q*v = b [v_1 v_2 .. v_q]
        prob_view = prob.view(pred_logit.shape[0], -1)
        prob_views = prob_view

        # print (f'++++++++++++++++prob.view(pred_logit.shape[0], -1): {prob.view(pred_logit.shape[0], -1)} ++++++++++++++++++++++')
        # print (f'++++++++++++++++prob_views.shape: {prob_views.shape} ++++++++++++++++++++++')
        topk_values, topk_indexes = torch.topk(prob_views, 100, dim=1)
        scores = topk_values
        # print (f'++++++++++++++++topk_values: {topk_values} ++++++++++++++++++++++')
        # print (f'++++++++++++++++topk_values.shape: {topk_values.shape} ++++++++++++++++++++++')
        # print (f'++++++++++++++++topk_indexes: {topk_indexes} ++++++++++++++++++++++')
        # print (f'++++++++++++++++topk_indexes.shape: {topk_indexes.shape} ++++++++++++++++++++++')

        # scores.append(topk_values*self.cost_feats_weight[feat_id])
        # scores = rearrange(scores, 'f b q -> f b q')
        # scores = reduce(scores, 'f b q -> b q', 'sum')
        # print(f'++++++++++++++++scores: {scores} ++++++++++++++++++++++')
        # print(f'++++++++++++++++scores.shape: {scores.shape} ++++++++++++++++++++++')

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # Todo 5a.1 -------------need to check the logic here too, since the outputs["pred_logits"] dim is diff now
        # outputs["pred_logits"].shape[3] = v
        # topk_boxes = q_i -> include label of b_i since we use share Voca
        # topk_boxes = torch.div(topk_indexes, outputs["pred_logits"].shape[3], rounding_mode='floor')
        topk_boxes = torch.div(topk_indexes, self.n_f1, rounding_mode='floor')
        # Label here include label of b_i since we use share Voca
        labels = topk_indexes % self.n_f1
        if self.dataset_file != 'coco_data':
            labels = labels.add(self.n_coor_bins)
        # if self.xywh:
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # print (f'++++++++++++++++results: {results} ++++++++++++++++++++++')
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, inter_output=0, prior_prob = 0.01):
        """
        set inter_output = -2 to return the last embedding befere the last linear layers. Which is token embed
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        layers =[]
        for n, k in zip([input_dim] + h, h + [output_dim]):
            linear = nn.Linear(n, k)
            layers.append(linear)
        layers[-1].bias.data = torch.ones(output_dim) * bias_value
        self.layers = nn.ModuleList(layers)
        self.inter_output = inter_output

    def forward(self, x):
        inter_x = []
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            inter_x.append(x)

        if self.inter_output == 0:
            return x
        else:
            return x, inter_x[self.inter_output]


def build_deformable_detr(args):
    if args.dataset_file == 'coco_data':
        num_classes = 91
        n_f1 = num_classes
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
        n_f1 = num_classes
    elif args.dataset_file == 'cityflow':
        voca = cityflow_build_voca(n_coor_bins=args.n_coor_bins, n_wh_bins=args.n_wh_bins)
        n_f1 = len(cityflow_feats.values()[1] + 1) # +1 for unknow class whose Id in voca is n_coor_bins (after b_i tokens)
        num_classes = 0
        for cityflow_feat in cityflow_feats.values():
            num_classes += len(cityflow_feat)
    else:
        num_classes = 20

    num_classes = num_classes + args.n_coor_bins if (args.seq_output) else num_classes
    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    # Todo 2 --------------------build main model DeformableDETR
    # Return  {'pred_logits': , 'pred_boxes': , 'aux_outputs', 'enc_outputs'}
    # out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_group_norm=args.num_group_norm,
        num_feat=args.num_feat,
        seq_output=args.seq_output,
        n_coor_bins=args.n_coor_bins,
        shared_voca = args.shared_voca,
        xywh = args.xywh
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # Todo 4 --------- compute criterion
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             cost_feats_weight=args.cost_feats_weight, shared_voca = args.shared_voca, n_coor_bins=args.n_coor_bins,
                             focal_gamma = args.focal_gamma)
    # criterion.to(device)

    #
    postprocessors = {'bbox': PostProcess(seq_output=args.seq_output, cost_feats_weight=args.cost_feats_weight,
                      shared_voca = args.shared_voca, n_coor_bins=args.n_coor_bins, n_f1=n_f1, dataset_file=args.dataset_file,
                      xywh=args.xywh)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
