# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List
import torch
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from pycocotools.coco import COCO
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from statistics import mean

from .deformable_detr import build_deformable_detr
from data_module import get_coco_api_from_dataset
from data_module.coco_data import build as build_coco_dataset

from data_module.coco_eval import CocoEvaluator
from util import misc, utils


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()

        # what is it ? => automatically save all the hyperparameters and log to checkpoint['hyper_parameters']
        # Model can also access to para by self.hparams.model_name
        # self.coco_evaluator = None
        self.save_hyperparameters()
        self.load_model()
        # self.configure_loss()

    # def forward(self, img):
    #     return self.model(img)

    def on_train_epoch_start(self):
        if self.hparams.cls_bbox_giou_coef is not None:
            pass
            # if self.current_epoch >


    def training_step(self, batch, batch_idx):
        # testing
        #  oringal [samples, targets = prefetcher.next()] in engine.py
        samples, targets = batch
        # Return  {'pred_logits': , 'pred_boxes': , 'aux_outputs', 'enc_outputs'}
        outputs = self.model(samples)
        # Todo 1 --------------------training_step --- targets
        train_loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        train_losses = sum(train_loss_dict[k] * weight_dict[k] for k in train_loss_dict.keys() if k in weight_dict)

        train_loss_dict.update({'train_weighted_sum_losses': train_losses})

        # self.log_dict(train_loss_dict, on_epoch=True, on_step=False, prog_bar=True, logger=True,
        #               batch_size=self.hparams.batch_size, sync_dist=True)



        for k, v in train_loss_dict.items():
            self.log(k, v, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size
                     , sync_dist=True)

        # metric_logger.update(grad_norm=grad_total_norm)
        return train_losses


    def on_validation_epoch_start(self):
        iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        dataset_val = build_coco_dataset(image_set='val', args=self.hparams)
        base_ds = get_coco_api_from_dataset(dataset_val)
        self.coco_evaluator = CocoEvaluator(base_ds, iou_types)

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        # targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Reduce by average the loss in each process
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        val_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        validation_stat_tmp = dict()
        validation_stat_tmp.update({'losses': val_losses})
        validation_stat_tmp.update(loss_dict_scaled)
        validation_stat_tmp.update(loss_dict_unscaled)
        validation_stat_tmp.update({'class_error': loss_dict['class_error']})

        validation_stat = {'val_'+key: value for key, value in validation_stat_tmp.items()}



        # Recover to original size
        # Todo 5.z ------------------ problem here in ge original size
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = [{'scores': s, 'labels': l, 'boxes': b}
        # Todo 5.a ------------------ problem here in postprecessor
        # Output of labels is wrong. 220. It must > 1000. Let check detailly
        # print (f'+++++++outputs: {outputs} +++++++++++++++++++++++++++++++++++')
        results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        # res =  {'image_id':{'scores': s, 'labels': l, 'boxes': b}}
        # print (f'+++++++targets: {targets} +++++++++++++++++++++++++++++++++++')
        # print (f'+++++++results: {results} +++++++++++++++++++++++++++++++++++')
        # print (f"+++++++target['image_id']: {targets[0]['image_id']} +++++++++++++++++++++++++++++++++++")
        # Todo 5.b ------------------ problem here in update result
        #  The target['labels'] is different, Seq have 2d tensor while no_Seq only have 1d tensor.
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(res)
        return validation_stat

    def validation_epoch_end(self, validation_stats):
        avg_stats = {}
        for key in set(validation_stats[0].keys()):
            # print (f'++++++++++key: {key}+++++++++++++++')
            avg_val = torch.stack([torch.tensor(x[key]) for x in validation_stats]).mean()
            avg_stats[key] = avg_val

        # Fix bug can't combine val acrross GPU
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if self.coco_evaluator is not None:
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()

        # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        # stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        coco_metrics_stat = self.coco_evaluator.coco_eval['bbox'].stats.tolist()
        coco_metrics_name = utils.coco_metrics_name()
        coco_metrics_dict = {k: v for k, v in zip(coco_metrics_name, coco_metrics_stat)}
        avg_stats.update(coco_metrics_dict)
        self.log_dict(avg_stats, on_epoch=True, on_step=False, prog_bar=True, logger=True,
                      batch_size=self.hparams.batch_size, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if not utils.match_name_keywords(n, self.hparams.lr_backbone_names
                                                      ) and not utils.match_name_keywords(n, self.hparams.lr_linear_proj_names) and p.requires_grad],
                "lr": self.hparams.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           utils.match_name_keywords(n, self.hparams.lr_backbone_names) and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           utils.match_name_keywords(n, self.hparams.lr_linear_proj_names) and p.requires_grad],
                "lr": self.hparams.lr * self.hparams.lr_linear_proj_mult,
            }
        ]

        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_dicts, lr=self.hparams.lr, momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_dicts, lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay)


        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
                


    def load_model(self):
        if self.hparams.model_name == 'deformable_detr':
            model, criterion, postprocessors = build_deformable_detr(self.hparams)

        self.model = model
        # loss fuction
        self.criterion = criterion
        # Possprocess output
        self.postprocessors = postprocessors

