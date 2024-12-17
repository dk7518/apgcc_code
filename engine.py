# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import logging 
import time
import shutil

import torch
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter

import util.misc as utils
from util.logger import vis, AvgerageMeter, EvaluateMeter

class Trainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, criterion):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.criterion = criterion

        self.logger = logging.getLogger('APGCC.train')
        self.log_period = cfg.SOLVER.LOG_FREQ
        self.eval_period = cfg.SOLVER.EVAL_FREQ
        self.output_dir = cfg.OUTPUT_DIR
        self.train_epoch = cfg.SOLVER.START_EPOCH
        self.batch_cnt = 0
        self.epochs = cfg.SOLVER.EPOCHS

        self.log_train = {"loss": AvgerageMeter()}
        for k in self.criterion.weight_dict.keys():
            self.log_train[k] = AvgerageMeter()
        self.log_eval = EvaluateMeter()
        self.best_models = []
        self.curr_time = time.time()
        self.writer = SummaryWriter(self.output_dir)
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.vis = cfg.VIS
        if self.vis:
            if not os.path.exists(os.path.join(self.output_dir, 'sample_result_for_train/')):
                os.makedirs(os.path.join(self.output_dir, 'sample_result_for_train/'))
            if not os.path.exists(os.path.join(self.output_dir, 'sample_result_for_val/')):
                os.makedirs(os.path.join(self.output_dir, 'sample_result_for_val/'))

        self.model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info('number of params:%d \n' % n_parameters)
        param_dicts = [
            {"params": [p for n, p in self.model_without_ddp.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if "encoder" in n and p.requires_grad],
                "lr": self.cfg.SOLVER.LR_BACKBONE,
            }]

        self.optimizer = torch.optim.Adam(param_dicts, lr=self.cfg.SOLVER.LR)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.SOLVER.LR_DROP)
       
        if self.cfg.MODEL.FROZEN_WEIGHTS is not None:
            checkpoint = torch.load(self.cfg.MODEL.FROZEN_WEIGHTS, map_location='cpu')
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])

        # if self.cfg.RESUME:
        #     checkpoint = torch.load(self.cfg.RESUME_PATH, map_location='cpu')
        #     self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        #     if not self.cfg.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #         self.optimizer.load_state_dict(checkpoint['optimizer'])
        #         self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #         self.cfg.SOLVER.START_EPOCH = checkpoint['epoch'] + 1
        #         self.train_epoch = checkpoint['epoch'] + 1
 # 체크포인트 로드 (RESUME)
        if self.cfg.RESUME:
            checkpoint = torch.load(self.cfg.RESUME_PATH, map_location='cpu')
            
            # 모델 상태 로드: 'model_state_dict' 또는 'model' 키 지원
            if 'model_state_dict' in checkpoint:
                self.model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("Loaded 'model_state_dict' from checkpoint.")
            elif 'model' in checkpoint:
                self.model_without_ddp.load_state_dict(checkpoint['model'])
                self.logger.info("Loaded 'model' from checkpoint.")
            else:
                raise KeyError("No valid model state found in checkpoint.")
            
            # 옵티마이저, 스케줄러, 에포크 정보 로드
            if not self.cfg.EVAL_MODE and 'optimizer_state_dict' in checkpoint and 'lr_scheduler_state_dict' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                self.cfg.SOLVER.START_EPOCH = checkpoint['epoch'] + 1
                self.train_epoch = checkpoint['epoch'] + 1
                self.best_metrics = checkpoint.get('best_metrics', None)
                self.logger.info(f"Resumed training from epoch {self.train_epoch}")
            else:
                self.train_epoch = self.cfg.SOLVER.START_EPOCH
                self.logger.info("Starting training from the specified start epoch.")

        #RESUME용 코드 
    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,  # 현재 에포크 저장
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    # def load_checkpoint(self, checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    #     self.best_metrics = checkpoint.get('best_metrics', None)
    #     start_epoch = checkpoint['epoch']
    #     self.logger.info(f"Checkpoint loaded: Resuming from epoch {start_epoch}")
    #     return start_epoch
   # Load checkpoint
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # 모델 상태 로드: 'model_state_dict' 또는 'model' 키 지원
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded 'model_state_dict' from checkpoint.")
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.logger.info("Loaded 'model' from checkpoint.")
        else:
            raise KeyError("No valid model state found in checkpoint.")
        
        # 옵티마이저 상태 로드
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Loaded 'optimizer_state_dict' from checkpoint.")
        
        # 스케줄러 상태 로드
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.logger.info("Loaded 'lr_scheduler_state_dict' from checkpoint.")
        
        # 기타 정보 로드
        self.best_metrics = checkpoint.get('best_metrics', None)
        start_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded: Resuming from epoch {start_epoch}")
        return start_epoch

    def handle_new_batch(self):
        self.batch_cnt += 1
        self.metric_logger.synchronize_between_processes()

    def handle_new_epoch(self):
        self.batch_cnt = 1
        self.lr_scheduler.step()

        stat = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        for k in self.log_train.keys():
            self.log_train[k].update(stat[k])

        if self.train_epoch % self.log_period == 0:
        
            logger_text = "[ep %d][lr %.7f][%.2fs]:"%(self.train_epoch, self.optimizer.param_groups[0]['lr'], time.time() - self.curr_time)
            for k in self.log_train.keys():
                logger_text += ' %s=%.8f/%.8f'%(k, stat[k], self.log_train[k].avg)

            self.logger.info('%s'%logger_text)
            for k in self.log_train.keys():
                self.writer.add_scalar('loss/%s'%k, stat[k], self.train_epoch)

        device = next(self.model.parameters()).device
        t1 = time.time()
        
        result = evaluate_crowd_counting(self.model, self.val_dl, device)
        t2 = time.time()
        self.log_eval.update(result[0], result[1], self.train_epoch)  # mae, mse, ep
        self.logger.info('[ep %d] Eval: MAE=%.6f/%.6f, MSE=%.6f/%.6f, Best[ep %d]: MAE=%.6f, MSE=%.6f, time:%.2fs  \n'
                                    %(self.train_epoch, 
                                      result[0], self.log_eval.MAE_avg, 
                                      result[1], self.log_eval.MSE_avg, 
                                      self.log_eval.best_ep, self.log_eval.MAE_min, self.log_eval.MSE_min, 
                                      t2 - t1))
        self.writer.add_scalar('metric/mae', result[0], self.train_epoch)
        self.writer.add_scalar('metric/mse', result[1], self.train_epoch)

        if abs(self.log_eval.MAE_min - result[0]) < 0.01:
                        self.save()
            
        checkpoint_latest_path = os.path.join(self.output_dir, 'latest.pth')
        torch.save({'model': self.model_without_ddp.state_dict()}, checkpoint_latest_path)
        # Save Checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_epoch_%d.pth' % self.train_epoch)
        checkpoint = {
            'epoch': self.train_epoch,
            'model_state_dict': self.model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'cfg': self.cfg,
        }
        # torch.save(checkpoint, checkpoint_path)
        # self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        # 기존 코드에서 'model' 키로 저장하고 있었다면, 이를 'model_state_dict'로 수정
        torch.save({'model_state_dict': self.model_without_ddp.state_dict()}, checkpoint_latest_path)
        self.logger.info(f"Latest checkpoint saved to {checkpoint_latest_path}")

        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.train_epoch += 1
        self.curr_time = time.time()

    def step(self, batch):
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        device = next(self.model.parameters()).device
        samples, targets = batch
        samples = samples.to(device)
        targets = [{k: v.to(device) if k!='name' else v for k, v in t.items()} for t in targets]
        # forward
        outputs = self.model(samples)

        # calc the losses
        loss_dict = self.criterion(outputs, targets, self.batch_cnt == 30)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        # backward
        losses.backward()
        if self.cfg.SOLVER.CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_MAX_NORM)
        
        self.optimizer.step()
        # update logger
        self.metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    # def save(self, ):
    #     checkpoint_best_path = os.path.join(self.output_dir, 'best_ep%d__%.5f_%.5f.pth' %\
    #                                         (self.log_eval.best_ep, self.log_eval.MAE_min, self.log_eval.MSE_min))
    #     torch.save(self.model_without_ddp.state_dict(), checkpoint_best_path)
    #     self.best_models.append(checkpoint_best_path)
    #     if len(best_models) > 10:
    #         if os.path.isfile(self.best_models[0]):
    #             os.remove(self.best_models[0])
    #         self.best_models.remove(self.best_models[0])

    #     shutil.copy(self.best_models[-1], os.path.join(self.output_dir, 'best.pth'))
    def save(self):
        # 체크포인트 경로 생성
        checkpoint_best_path = os.path.join(
            self.output_dir,
            'best_ep%d__%.5f_%.5f.pth' % (
                self.log_eval.best_ep, 
                self.log_eval.MAE_min, 
                self.log_eval.MSE_min
            )
        )
        
        # 모델 상태 저장
        torch.save(self.model_without_ddp.state_dict(), checkpoint_best_path)
        
        # best_models 리스트에 추가
        self.best_models.append(checkpoint_best_path)

        # best_models의 길이가 10을 초과하면 가장 오래된 파일 삭제
        if len(self.best_models) > 10:
            if os.path.isfile(self.best_models[0]):
                os.remove(self.best_models[0])
            self.best_models.pop(0)  # 리스트에서 첫 번째 항목 제거

        # 최신 최적의 모델을 best.pth로 복사
        shutil.copy(self.best_models[-1], os.path.join(self.output_dir, 'best.pth'))
# the inference routine
@torch.no_grad()
def evaluate_crowd_counting(model, data_loader, device, threshold=0.5, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            
            outputs_points = outputs['pred_points'][0]
            gt_cnt = targets[0]['point'].shape[0]

            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())
            # if specified, save the visualized images
            if vis_dir is not None: 
                vis(samples, targets, [points], vis_dir)
            # accumulate MAE, MSE
            mae = abs(predict_cnt - gt_cnt)
            mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    return mae, mse

@torch.no_grad()
def evaluate_crowd_counting_and_loc(model, data_loader, device, threshold=0.5, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    nMAE = 0
    intervals = {}
    tp_sum_4 = 0
    gt_sum = 0
    et_sum = 0
    tp_sum_8 = 0
    for ct, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        # if specified, save the visualized images
        if vis_dir is not None: 
            vis(samples, targets, [points], vis_dir)

        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))

        # nMAE += mae/gt_cnt
        interval = int(gt_cnt / 250)
        if interval not in intervals:
            intervals[interval] = [mae/gt_cnt]
        else:
            intervals[interval].append(mae/gt_cnt)

        tp_4 = utils.compute_tp(points, targets[0]['point'], 4)
        tp_8 = utils.compute_tp(points, targets[0]['point'], 8)
        tp_sum_4 += tp_4
        gt_sum += gt_cnt
        et_sum += predict_cnt
        tp_sum_8 += tp_8

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    # nMAE /= len(data_loader)
    ap_4 = tp_sum_4 / float(et_sum+1e-10)
    ar_4 = tp_sum_4 / float(gt_sum+1e-10)
    f1_4 = 2*ap_4*ar_4 / (ap_4+ar_4+1e-10)
    ap_8 = tp_sum_8 / float(et_sum+1e-10)
    ar_8 = tp_sum_8 / float(gt_sum+1e-10)
    f1_8 = 2*ap_8*ar_8 / (ap_8+ar_8+1e-10)
    local_result = {'ap_4': ap_4, 'ar_4':ar_4, 'f1_4':f1_4, 'ap_8': ap_8, 'ar_8':ar_8, 'f1_8':f1_8}
    return mae, mse, local_result
