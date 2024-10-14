import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

from timm.data import Mixup, create_transform
from timm.utils import accuracy, AverageMeter
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch import optim as optim

from torchvision import datasets, transforms

from thop import profile, clever_format

import models
from utils import check_keywords_in_name, create_logger, NativeScalerWithGradNormCount, reduce_tensor

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

# 指定哪几张GPU进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

def get_args_parser():
    parser = argparse.ArgumentParser("Models for image classification", add_help=False)
    # --------------------------------------------------------------------------------
    # 数据集参数设置
    # --------------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for a single GPU")  # Depend on the task
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")  # Depend on the task
    parser.add_argument('--img_size', type=int, default=224, help="Input image size")
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help="Interpolation to resize image (random, bilinear, bicubic)")
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loading threads")
    # --------------------------------------------------------------------------------
    # 模型参数设置
    # --------------------------------------------------------------------------------
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name')
    parser.add_argument('--resume', type=str, default='', help="Checkpoint to resume")
    parser.add_argument('--load_ckpt', type=str, default='', help="Load the pretrained ckpt")
    parser.add_argument('--num_classes', type=int, default=1000, help="Number of classes")  # Depend on the task
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing")  #
    # --------------------------------------------------------------------------------
    # 训练参数设置
    # --------------------------------------------------------------------------------
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)  # Depend on the task
    parser.add_argument('--warmup_epochs', type=int, default=5)  # Depend on the task
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--warmup_lr', type=float, default=2e-7)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--clip_grad', type=float, default=5.0, help="Clip gradient norm")
    parser.add_argument('--auto_resume', type=bool, default=True, help="Auto resume from latest checkpoint")
    parser.add_argument('--accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    # 学习率调整策略参数设置
    parser.add_argument('--lr_scheduler_name', type=str, default='cosine')
    parser.add_argument('--decay_epochs', type=int, default=30,  # Depend on the task
                        help="Epoch interval to decay LR, used in StepLRScheduler")
    parser.add_argument('--decay_rate', type=float, default=0.1, help="LR decay rate, used in StepLRScheduler")
    parser.add_argument('--warmup_prefix', type=bool, default=False, help="warmup_prefix used in CosineLRScheduler")
    # Optimizer参数设置
    parser.add_argument('--optim_name', type=str, default='adamw')
    parser.add_argument('--optim_eps', type=float, default=1e-8, help="Optimizer Epsilon")
    parser.add_argument('--optim_betas', type=tuple, default=(0.9, 0.999), help="Optimizer Betas")
    parser.add_argument('--optim_momentum', type=float, default=0.9, help="SGD momentum")
    # --------------------------------------------------------------------------------
    # 数据增强参数设置
    # --------------------------------------------------------------------------------
    parser.add_argument('--color_jitter', type=float, default=0.4, help="Color jitter factor")
    parser.add_argument('--auto_augment', type=str, default='rand-m9-mstd0.5-inc1',
                        help="Use AutoAugment policy. 'v0' or 'original'")
    parser.add_argument('--reprob', type=float, default=0.25, help="Random erase prob")  # *
    parser.add_argument('--remode', type=str, default='pixel', help="Random erase mode")
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--mixup', type=float, default=0.8, help="Mixup alpha, mixup enabled if > 0")  # *
    parser.add_argument('--cutmix', type=float, default=1.0, help="Cutmix alpha, cutmix enabled if > 0")
    parser.add_argument('--cutmix_minmax', type=float, default=None,
                        help="Cutmix min/max ratio, overrides alpha and enables cutmix if set")
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help="Probability of performing mixup or cutmix when either/both is enabled")
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help="Probability of seitching to cutmix when both mixup and cutmix enabled")
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help="How to apply mixup/cutmix params. Per 'batch', 'pair', or 'elem'")
    # --------------------------------------------------------------------------------
    # 其他参数设置
    # --------------------------------------------------------------------------------
    parser.add_argument('--amp_enable', type=bool, default=True, help="Enable Pytorch automatic mixed precision (amp)")
    parser.add_argument('--output', type=str, default='output', help="Path to output folder")
    parser.add_argument('--tag', type=str, default='default', help="Tag of experiment")
    parser.add_argument('--save_freq', type=int, default=10, help="Frequency to save checkpoint")
    parser.add_argument('--print_freq', type=int, default=20, help="Frequency to logging info")
    parser.add_argument('--seed', type=int, default=16, help="Fixed random seed")
    parser.add_argument('--throughput_mode', type=bool, default=False, help="Test throughput only")
    parser.add_argument('--local_rank', type=int, required=True, help="Local rank for DistributedDataParallel")

    args, unparsed = parser.parse_known_args()

    return args


def main(args):

    # ========================= 加载数据集 =========================
    # 数据转换和增强
    # color_jitter: 在一定范围内, 对图像的亮度(Brightness), 对比度(Contrast), 饱和度(Saturation)和色相(Hue)进行随机变换
    # auto_augment: 自动数据增强策略
    train_transform = create_transform(
        input_size=args.img_size, is_training=True,
        color_jitter=args.color_jitter if args.color_jitter > 0 else None,
        auto_augment=args.auto_augment if args.auto_augment != 'none' else None,
        re_prob=args.reprob, re_mode=args.remode,
        re_count=args.recount, interpolation=args.interpolation
    )
    size = int((256 / 224) * args.img_size)
    val_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=_pil_interp(args.interpolation)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    # 构建数据集
    train_dataset = datasets.ImageFolder(args.data_path + '/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(args.data_path + '/val', transform=val_transforms)
    # 构建分布式训练的数据采样器(sampler)
    num_tasks = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        dataset=train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
    )
    val_sampler = torch.utils.data.DistributedSampler(dataset=val_dataset, shuffle=False)
    # 构建数据加载器(dataloader)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True  # drop_last: 是否舍去最后一个batch的数据
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, sampler=val_sampler, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    # ========================= MixUp数据增强 =========================
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes
        )

    # ========================= 创建模型 =========================
    # 根据所选的模型(model_name)来创建模型
    logger.info(f"Creating model:{args.model_name}")
    model = models.__dict__[args.model_name](num_classes=args.num_classes)
    logger.info(f"Using model: {args.model_name}")
    # 计算模型的FLOPs与参数量
    flops, params = profile(model, inputs=(torch.rand([1, 3, 224, 224]), ))
    flops, params = clever_format([flops, params], "%.2f")
    logger.info(f"The number of parameters: {params}")
    logger.info(f"number of GFLOPs: {flops}")
    # 开启SyncBN, 将模型中的BN替换成SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 将模型加载到GPU
    model.cuda()
    # 先将模型保存到model_without_ddp
    # 使用DDP后, 原模型已经被封装, 运行时进行发布
    # 对于模型的保存, 要么先将模型保存到model_without_ddp, 保存时保存model_without_ddp; 要么就在DDP后, 保存model.module模块
    # 读取checkpoint也需将checkpoint保存到model_without_ddp
    model_without_ddp = model

    # ========================= 构建Optimizer =========================
    # 获取模型的学习参数, 并进行分组, 设置weight decay, 而Norm层, bias层无需设置weight decay
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    parameters = [{'params': has_decay},
                  {'params': no_decay, 'weight_decay': 0.}]
    # 设置Optimizer参数
    opt_lower = args.optim_name.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=args.optim_momentum, nesterov=True,
                              lr=args.base_lr, weight_decay=args.weight_decay)
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=args.optim_eps, betas=args.optim_betas,
                                lr=args.base_lr, weight_decay=args.weight_decay)

    # ========================= 分布式训练(DDP) =========================
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False,
                                                      find_unused_parameters=True)

    # ========================= Loss Scaler =========================
    loss_scaler = NativeScalerWithGradNormCount()  # 用于自动混合精度

    # ========================= 构建LR Scheduler =========================
    if args.accumulation_steps > 1:
        n_iter_per_epoch = len(train_loader) // args.accumulation_steps
    else:
        n_iter_per_epoch = len(train_loader)
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
    decay_steps = int(args.decay_epochs * n_iter_per_epoch)
    lr_scheduler = None
    if args.lr_scheduler_name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if args.warmup_prefix else num_steps,
            t_mul=1.,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=args.warmup_prefix
        )
    elif args.lr_scheduler_name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False
        )

    # ========================= 定义损失函数 =========================
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    # ========================= 加载预训练权重 =========================
    if args.load_ckpt:
        logger.info(f"===============> Load ckpt from {args.load_ckpt}...............")
        checkpoint = torch.load(args.load_ckpt, map_location='cpu')
        # 读取checkpoint也需将模型保存到model_without_ddp
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint, strict=False)
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"==> Loaded successfully '{args.load_ckpt}')")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        del checkpoint
        torch.cuda.empty_cache()
        acc1, acc5, loss = validate(args, val_loader, model, args.start_epoch - 1)
        logger.info(f"Accuracy of the model on the {len(val_dataset)} val images: {acc1:.1f}%")

    # ========================= 断点续训 =========================
    if args.auto_resume:
        checkpoints = os.listdir(args.output)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
        logger.info(f"All checkpoints founded in {args.output} : {checkpoints}")
        if len(checkpoints) > 0:
            latest_checkpoint = max([os.path.join(args.output, ckpt) for ckpt in checkpoints], key=os.path.getmtime)
            logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
            resume_file = latest_checkpoint
        else:
            resume_file = None
        if resume_file:
            args.resume = resume_file
            logger.info(f"Auto resuming from {resume_file}")
        else:
            logger.info(f"No checkpoint found in {args.output}, ignoring auto resume")
    if args.resume:
        logger.info(f"===============> Resuming from {args.resume}...............")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 读取checkpoint也需将模型保存到model_without_ddp
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"==> Loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        del checkpoint
        torch.cuda.empty_cache()
        acc1, acc5, loss = validate(args, val_loader, model, args.start_epoch-1)
        logger.info(f"Accuracy of the model on the {len(val_dataset)} val images: {acc1:.1f}%")

    # ========================= 测试模型吞吐量 =========================
    if args.throughput_mode:
        throughput(val_loader, model, logger)
        return

    # ========================= 开始训练 =========================
    logger.info("====================> Start training! <====================")
    start_time = time.time()  # 开始训练的时刻
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(args, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        # 需要将模型保存在主进程,否则在使用多卡时无法load模型
        if torch.distributed.get_rank() == 0 and ((epoch+1) % args.save_freq == 0) or epoch == (args.epochs - 1):
            save_state = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'max_accuracy': max_accuracy,
                          'scaler': loss_scaler.state_dict(),
                          'epoch': epoch}
            save_path = os.path.join(args.output, f"ckpt_epoch_{epoch}.pth")
            logger.info(f"====================> {save_path} saving..........")
            torch.save(save_state, save_path)
            logger.info(f"====================> {save_path} saved !!!")

        # evaluate on validation set
        acc1, acc5, loss = validate(args, val_loader, model, epoch)
        logger.info(f"Accuracy of the model on the {len(val_dataset)} val images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"So far, the max accuracy: {max_accuracy:.2f}%")

        # 保存每轮及精度最高的模型
        save_state = {'model': model_without_ddp.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'max_accuracy': max_accuracy,
                      'scaler': loss_scaler.state_dict(),
                      'epoch': epoch}
        latest_model_save_path = os.path.join(args.output, 'model-latest.pth')
        best_model_save_path = os.path.join(args.output, 'model-best.pth')
        # 需要将模型保存在主进程,否则在使用多卡时无法load模型
        if torch.distributed.get_rank() == 0 and acc1 == max_accuracy:
            torch.save(save_state, best_model_save_path)
        if torch.distributed.get_rank() == 0:
            torch.save(save_state, latest_model_save_path)

    total_time = time.time() - start_time  # 训练总时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"====================> Training completed! <====================")
    logger.info(f"Training time: {total_time_str}")


def train_one_epoch(args, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()  # 梯度清零

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()  # 开始训练某一轮的时刻
    end = time.time()  # 开始训练某个batch的时刻
    for step, (images, labels) in enumerate(train_loader):
        # 将数据加载到GPU
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # 计算损失
        with torch.cuda.amp.autocast(enabled=args.amp_enable):
            outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / args.accumulation_steps

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),
            create_graph=is_second_order, update_grad=(step + 1) % args.accumulation_steps == 0
        )
        if (step + 1) % args.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + step) // args.accumulation_steps)
        loss_scaler_value = loss_scaler.state_dict()['scale']

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), labels.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scaler_value)
        batch_time.update(time.time() - end)  # 训练该个batch所花费的时间
        end = time.time()  # 重置开始训练某个batch的时刻

        if step % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - step)
            logger.info(f"Train:[{epoch}/{args.epochs}][{step}/{num_steps}] "
                        f"Eta:{datetime.timedelta(seconds=int(etas))} "
                        f"Lr:{lr:.6f} "
                        f"Loss:{loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                        f"Grad_norm:{norm_meter.val:.4f}({norm_meter.avg:.4f}) "
                        f"Mem：{memory_used:.0f}MB")
    epoch_time = time.time() - start  # 训练一个epoch所需的时间
    logger.info(f"====================> Epoch {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(args, val_loader, model, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()  # 开始验证某个batch的时刻
    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp_enable):
            output = model(images)

        # 计算精度和损失
        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))
        acc5_meter.update(acc5.item(), labels.size(0))

        # 计算验证一个batch所需的时间
        batch_time.update(time.time() - end)
        end = time.time()  # 充值开始验证某个batch的时刻

        if step % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f"Test:[{step}/{len(val_loader)}] "
                        f"Time:{batch_time.val:.3f}({batch_time.avg:.3f}) "
                        f"Loss:{loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                        f"Acc@1:{acc1_meter.val:.3f}({acc1_meter.avg:.3f}) "
                        f"Acc@5:{acc5_meter.val:.3f}({acc5_meter.avg:.3f}) "
                        f"Mem:{memory_used:.0f}MB")
    logger.info(f"====================> After epoch {epoch}, Acc@1: {acc1_meter.avg:.3f}, Acc@5: {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args = get_args_parser()

    # ========================= 分布式训练初始化 =========================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    # windows系统只支持gloo, 在linux系统上推荐使用nccl
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()  # 实现不同进程之间的数据同步

    # ========================= 设置随机种子 =========================
    seed = args.seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True  # 让cudnn内置的auto-tuner自动寻找最适合当前配置的高效算法, 优化运行效率

    # 模型输出的保存位置
    args.output = os.path.join(args.output, args.model_name, args.tag)
    os.makedirs(args.output, exist_ok=True)
    # 创建日志器, 记录训练过程
    logger = create_logger(output_dir=args.output, dist_rank=torch.distributed.get_rank(), name=f"{args.model_name}")

    main(args)
