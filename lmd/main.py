import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
# from pytorch_lightning.cli import LightningCLI

# from .deformable_detr.model.model_interface import MInterface
from data_module.data_interface import DInterface
from model.model_interface import MInterface
from util.utils import load_model_path_by_args
import copy


def load_callbacks():
    callbacks = []

    # callbacks.append(pl.loggers.TensorBoardLogger(
    #     save_dir=args.logger_dir,
    #     name="model"))

    # callbacks.append(plc.EarlyStopping(
    #      monitor='val_acc',
    #      mode='max',
    #      patience=10,
    #      min_delta=0.001
    #  ))

    if args.save_checkpoint:
        callbacks.append(plc.ModelCheckpoint(
            dirpath=args.checkpoints_path,
            monitor="AP@IoU_0.50_0.95",
            filename='deformableDetr-{epoch:02d}-{AP@IoU_0.50_0.95:.3f}',
            every_n_epochs=2,  # do not save 2 consecutive epoch
            save_top_k=0,
            mode='max',
            save_last=True,
            save_on_train_epoch_end=True
        ))

    # if args.strategy_training == 'ddp':
    #     callbacks.append(pl.strategies.DDPStrategy(
    #         find_unused_parameters=False))
    # elif args.strategy_training == 'deepspeed2':
    #     callbacks.append(pl.strategies.DeepSpeedStrategy(
    #         stage=2))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    callbacks.append(plc.GradientAccumulationScheduler(scheduling=args.accumulate_batches))

    callbacks.append(plc.RichModelSummary(max_depth=args.model_summary_deep))

    callbacks.append(plc.RichProgressBar())

    if args.swa != 0:
        callbacks.append(plc.StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy = args.annealing_strategy
        ))

    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        # args.resume_from_checkpoint = load_path
        # print (f'+++++++++++++++++++++Loading model from checkpoints: {load_path}++++++++++++++++++++++++++++++++')

    # https: // pytorch - lightning.readthedocs.io / en / latest / visualize / logging_intermediate.html
    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='', name=args.log_dir)
    args.callbacks = load_callbacks()
    # args.logger = logger
    # print ('')

    trainer = Trainer.from_argparse_args(args)
    # Turn on tune model (should run
    if args.auto_find_lr:
        args_without_ddp = copy.deepcopy(args)
        args_without_ddp.strategy = None
        args_without_ddp.devices = [args.devices[-1]]
        # args_without_ddp.accumulate_grad_batches = None
        trainer_without_ddp = Trainer.from_argparse_args(args_without_ddp)
        lrfinder = trainer_without_ddp.tuner.lr_find(model, datamodule=data_module)
        lr = lrfinder.suggestion()
        print(f'++++++++++++++++++++++++lr {lr}++++++++++++++++++++++++++++++++++++++++')
        args.lr = lr
        args.lr_backbone = args.lr_backbone * args.lr
        trainer = None


    if args.val_mode:
        trainer.validate(model, data_module, ckpt_path=load_path)
    else:
        # Fit = train + val + test
        trainer.fit(model, data_module, ckpt_path=load_path)

    # cli = LightningCLI(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Deformable DETR Detector', add_help=False)
    # parser = Trainer.add_argparse_args(parser)

    # Special configure
    parser.add_argument('--fast_dev_run_all', action='store_true')  # if running fast for testing
    parser.add_argument('--cpu_testing', action='store_true')  # if testing code in cpu
    parser.add_argument('--aic21_track3_extract', action='store_true')  # if extract image only
    parser.add_argument('--tune_model', action='store_true')
    parser.add_argument('--val_mode', action='store_true')
    parser.add_argument('--swa', default=0, type=int)


    # Basic - for all model
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices_', default=None, type=str)  # '2' run 2 first gpus - '[2]' gpu number 2
    # parser.add_argument('--devices_', default='[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]', type=str)  # '2' run 2 first gpus - '[2]' gpu number 2
    # parser.add_argument('--devices_', default='15', type=str) # '2' run 2 first gpus - '[2]' gpu number 2
    # parser.add_argument('--auto_select_gpus', default=True, type=bool)
    parser.add_argument('--gradient_clip_val', default=0.1, type=float)  # gradient clipping
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--num_sanity_val_steps', default=0, type=int)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--profiler', default='simple', type=str)
    parser.add_argument('--precision', default=32, type=int)
    # parser.add_argument('--strategy', default=strategies.DDPStrategy(find_unused_parameters=False))
    parser.add_argument('--strategy', default='ddp', choices=['ddp_find_unused_parameters_false', 'deepspeed_stage_2'],
                        type=str)
    # parser.add_argument('--accumulate_grad_batches', default={0:4, 40:40})
    # parser.add_argument('--accumulate_grad_batches', default=7, type=int)
    parser.add_argument('--accumulate_batches', default='0_7_40_70', type=str)
    parser.add_argument('--model_summary_deep', default=2, type=int)

    # Training Control
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--seed', default=1709, type=int)
    parser.add_argument('--optimizer', default='adamw', const='adamw', nargs='?', choices=['adamw', 'sgd'], type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--number_using_data', default=-1, type=int)  # How many percent of data to be used

    # LR Scheduler
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_scheduler', default='step', const='step', nargs='?', choices=['step', 'cosine', None], type=str)
    parser.add_argument('--lr_decay_steps', default=100, type=int)  # lr_drop
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)  # use for cosine lr_scheduler

    # Output - Logging
    # parser.add_argument('--default_root_dir ', default='experiment', type=str)
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--save_checkpoint', default=True, type=bool)  # not save checkpoint ?
    parser.add_argument('--checkpoints_path', default=None,
                        type=str)  # Create save save checkpoints in dir different to log_dir
    parser.add_argument('--log_every_n_steps ', default=50, type=int)

    # Tuning model
    # parser.add_argument('--auto_scale_batch_size', default='binsearch', type=str) #Can't use with different-size inputs
    parser.add_argument('--auto_find_lr', action='store_true')
    #  Set benchmark to true to improve performance. But can't reproduce (see torch.backends.cudnn.benchmark)
    #  increase the speed of your system if your input sizes don’t change. However, if they do, your system is slower
    parser.add_argument('--benchmark', action='store_true')
    # Ensure full reproducibility - do not turn on if use benchmark
    # parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false')

    # ----------------------------------------------------------
    # DeformableDETR
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco_data', type=str)
    parser.add_argument('--data_dir', default='../data/coco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # Training control
    parser.add_argument('--model_name', default='deformable_detr', type=str)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=0.1, type=float, help='ratio * args.lr')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_xywh', dest='xywh', action='store_false',
                        help="Disables xywh and use xyxy")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--cls_bbox_giou_coef', default=None, type=str,
                        help='loss weight by epoch exp: 0:7_0_0,20:5_2_2')
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2, type=float)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of scale-feature levels')

    # * sequence output
    parser.add_argument('--no_seq_output', dest='seq_output', action='store_false',
                        help="Out put as sequence of tokens")
    parser.add_argument('--num_feat', default=5, type=int,
                        help="number of featture. cityflow is 6 [b1 b2 b3 b4 car_type car_col]")
    parser.add_argument('--cost_feats_weight', default='1_1_1_1_2', type=str, # [0.5,0.5] if cityflow_data
                        help="weightLoss for multi feat, sum must be 1")
    parser.add_argument('--n_coor_bins', default=1000, type=int,
                        help="number of bins for x,y coordinate of bbox")
    parser.add_argument('--n_wh_bins', default=1000, type=int,
                        help="number of bins for w,h - weight and height of bbox")
    parser.add_argument('--no_shared_voca', dest='shared_voca', action='store_false',
                        help="Use shared voca for b_i and f_i")

    # * Transformer
    parser.add_argument('--num_group_norm', default=32, type=int,
                        help="number of groups to apply Group Normalization, see CLASStorch.nn.GroupNorm")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Turn on to add other parameter to parser - However get error of duplication parameter
    # parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # # Devices transfer from parser
    device_text = args.devices_
    if '[' not in device_text:
        args.devices = list(range(int(device_text)))
    else:
        args.devices = [int(x) for x in device_text[1:-1].split(',')]

    #
    tmp = [int(i) for i in args.accumulate_batches.split('_')]
    args.accumulate_batches = {tmp[i]:tmp[i+1] for i in range(0, len(tmp), 2)}
    print(f'+++++++++++++++++++++args.accumulate_batches {args.accumulate_batches}++++++++++++++++++++++++++')

    # lr_backbone
    args.lr_backbone = args.lr_backbone * args.lr
    # if args.seq_output:
    #     print (f'+++++++++++++++checking sequence output+++++++++++++')
    # Change dimension of multiplier of number of feat => may used later for split seq
    # args.dim_feedforward = args.dim_feedforward // args.num_feat * args.num_feat
    # args.hidden_dim = args.hidden_dim // args.num_feat * args.num_feat

    if args.cls_bbox_giou_coef is not None:
        pass

    # # Update matching loss as training loss
    args.set_cost_class = args.cls_loss_coef
    args.set_cost_bbox = args.bbox_loss_coef
    args.set_cost_giou = args.giou_loss_coef

    args.cost_feats_weight = [int(i) for i in args.cost_feats_weight.split('_')]
    args.cost_feats_weight = [float(i) / sum(args.cost_feats_weight) for i in args.cost_feats_weight]

    if args.dataset_file == 'cityflow_data':
        args.seq_output = True

    if not args.seq_output:
        args.cost_feats_weight = [1]

    # CPU running test code
    if args.fast_dev_run_all:
        # args.fast_dev_run = True
        # args.log_every_n_steps = 2
        args.number_using_data = 100

    if args.cpu_testing:
        args.accelerator = 'cpu'
        args.devices = 6
        args.num_workers = 6

        args.dim_feedforward = 32
        args.hidden_dim = 8
        args.num_group_norm = 2
        args.number_using_data = 10

    if args.aic21_track3_extract:
        """Extract data only
        """
        args.data_dir = '../data/AIC21_Track3_MTMC_Tracking'
        args.mapper = '../data/AIC21_Track3_MTMC_Tracking/mapper_nls2Track3.json'
        args.extract_dir = 'bbox_nl'
        # args.dataset_file = 'stop_after_preparedata'

    if args.swa != 0:
        """
        Update parameter for swa
        """
        args.swa_lrs = args.lr*0.1
        args.swa_epoch_start = copy.deepcopy(args.max_epochs)
        args.annealing_epochs = args.swa
        args.max_epochs = args.swa_epoch_start + args.annealing_epochs
        args.annealing_strategy = 'linear'
    # List Arguments
    # args.mean_sen = [0.485, 0.456, 0.406]
    # args.std_sen = [0.229, 0.224, 0.225]

    main(args)
