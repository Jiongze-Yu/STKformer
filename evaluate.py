import os
import time
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from thop import profile, clever_format
import models
from utils import NativeScalerWithGradNormCount

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
except ImportError:
    from timm.data.transforms import _pil_interp

# TensorRT imports
#import tensorrt as trt
#import pycuda.driver as cuda
#import pycuda.autoinit

# 指定GPU进行评估
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args_parser():
    parser = argparse.ArgumentParser("Models for image evaluation", add_help=False)
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for a single GPU")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--img_size', type=int, default=224, help="Input image size")
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help="Interpolation to resize image (random, bilinear, bicubic)")
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of data loading threads")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name')
    parser.add_argument('--resume', type=str, default='', help="Checkpoint to resume")
    parser.add_argument('--load_ckpt', type=str, required=True, help="Load the pretrained ckpt or TRT model")
    parser.add_argument('--num_classes', type=int, default=1000, help="Number of classes")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing")
    parser.add_argument('--amp_enable', type=bool, default=True, help="Enable Pytorch automatic mixed precision (amp)")
    parser.add_argument('--output', type=str, default='output', help="Path to output folder")
    # parser.add_argument('--tag', type=str, default='default', help="Tag of experiment")
    parser.add_argument('--print_freq', type=int, default=20, help="Frequency to logging info")
    parser.add_argument('--seed', type=int, default=16, help="Fixed random seed")
    parser.add_argument('--throughput_mode', type=bool, default=False, help="Test throughput only")

    args, unparsed = parser.parse_known_args()
    return args

def load_pytorch_model(args):
    print(f"Creating model: {args.model_name}")
    model = models.__dict__[args.model_name](num_classes=args.num_classes)
    print(f"Using model: {args.model_name}")
    flops, params = profile(model, inputs=(torch.rand([1, 3, 224, 224]), ))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"The number of parameters: {params}")
    print(f"Number of GFLOPs: {flops}")
    model.cuda()

    if args.load_ckpt:
        print(f"===============> Load ckpt from {args.load_ckpt}...............")
        checkpoint = torch.load(args.load_ckpt, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"==> Loaded successfully '{args.load_ckpt}'")
        del checkpoint
        torch.cuda.empty_cache()

    return model

def load_tensorrt_model(args):
    print(f"Loading TensorRT engine from {args.load_ckpt}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.load_ckpt, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    return engine, context

def main(args):

    size = int((256 / 224) * args.img_size)
    val_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=_pil_interp(args.interpolation)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    if args.load_ckpt.endswith('.pth'):
        model = load_pytorch_model(args)
        model_type = "pytorch"
    elif args.load_ckpt.endswith('.trt'):
        engine, context = load_tensorrt_model(args)
        model_type = "tensorrt"
    else:
        raise ValueError("Unsupported model file extension. Please use '.pth' for PyTorch models or '.trt' for TensorRT models.")

    if model_type == "pytorch":
        model.eval()
        acc1, acc5, loss = validate(args, val_loader, model, args.start_epoch - 1)
        print(f"Accuracy of the model on the {len(val_dataset)} val images: {acc1:.1f}%")

    elif model_type == "tensorrt":
        # TensorRT inference setup
        def allocate_buffers(engine):
            inputs = []
            outputs = []
            bindings = []
            stream = cuda.Stream()
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem})
            return inputs, outputs, bindings, stream

        def do_inference_v2(context, bindings, inputs, outputs, stream, batch_size=1):
            # Set input shapes if needed
            context.set_binding_shape(0, (batch_size, 3, args.img_size, args.img_size))
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
            # Run inference using execute_async_v2.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out['host'] for out in outputs]

        # Allocate buffers for input and output
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        # Validate the TensorRT model
        correct1 = 0
        correct5 = 0
        total = 0
        inference_time = 0
        total_images = 0

        for step, (images, labels) in enumerate(val_loader):
            images = images.numpy()
            np.copyto(inputs[0]['host'], images.ravel())

            # Measure inference time
            t0 = time.time()
            outputs_host = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t1 = time.time()

            batch_time = t1 - t0
            inference_time += batch_time
            total_images += images.shape[0]

            # Get the output as a numpy array
            output = np.array(outputs_host[0]).reshape(args.batch_size, -1)

            # Calculate top-1 and top-5 accuracy
            top1_pred = np.argmax(output, axis=1)
            top5_pred = np.argsort(output, axis=1)[:, -5:]

            correct1 += np.sum(top1_pred == labels.numpy())
            correct5 += np.sum([1 if label in top5 else 0 for label, top5 in zip(labels.numpy(), top5_pred)])
            total += labels.size(0)

            if step % args.print_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                FPS = total_images / inference_time
                print(f"Batch [{step}/{len(val_loader)}]: "
                      f"Top-1 accuracy: {correct1 / total * 100:.2f}%, "
                      f"Top-5 accuracy: {correct5 / total * 100:.2f}%, "
                      f"Mem:{memory_used:.0f}MB, "
                      f"FPS: {FPS:.2f}")

        FPS = total_images / inference_time
        print(
            f"Final accuracy: Top-1: {correct1 / total * 100:.2f}%, Top-5: {correct5 / total * 100:.2f}%, FPS: {FPS:.2f}")

@torch.no_grad()
def validate(args, val_loader, model, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    inference_time = 0
    sample_num = 0
    start_num = 200
    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute output
        if sample_num>=start_num:
            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=args.amp_enable):
                output = model(images)
            inference_time += time.time() - t0
            # print(time.time() - t0, sample_num)
        else:
            with torch.cuda.amp.autocast(enabled=args.amp_enable):
                output = model(images)
        sample_num += images.shape[0]

        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))
        acc5_meter.update(acc5.item(), labels.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if sample_num>=start_num:
                FPS = (sample_num-start_num) / inference_time
                print(f"Test:[{step}/{len(val_loader)}] "
                      f"Time:{batch_time.val:.3f}({batch_time.avg:.3f}) "
                      f"Loss:{loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                      f"Acc@1:{acc1_meter.val:.3f}({acc1_meter.avg:.3f}) "
                      f"Acc@5:{acc5_meter.val:.3f}({acc5_meter.avg:.3f}) "
                      f"Mem:{memory_used:.0f}MB "
                      f"FPS: {FPS:.2f}")
            else:
                print(f"Test:[{step}/{len(val_loader)}] "
                      f"Time:{batch_time.val:.3f}({batch_time.avg:.3f}) "
                      f"Loss:{loss_meter.val:.4f}({loss_meter.avg:.4f}) "
                      f"Acc@1:{acc1_meter.val:.3f}({acc1_meter.avg:.3f}) "
                      f"Acc@5:{acc5_meter.val:.3f}({acc5_meter.avg:.3f}) "
                      f"Mem:{memory_used:.0f}MB")
    print(f"====================> After epoch {epoch}, Acc@1: {acc1_meter.avg:.3f}, Acc@5: {acc5_meter.avg:.3f}, FPS: {FPS:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range (50):
            model(images)
        torch.cuda.synchronize()
        print(f"Throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"Batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    args = get_args_parser()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # args.output = os.path.join(args.output, args.model_name, args.tag)
    # os.makedirs(args.output, exist_ok=True)

    main(args)

# python evaluate.py --batch_size 1 --model_name STKformer_0_75_100_25 --data_path E:\working\科研\Smoke_Detection\program\dataset\new-smoke-fog-dataset --load_ckpt ./weights/1_DSDF/model-best.pth
