import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict
from collections import deque
from typing import List
from typing import Optional
import torchvision
import numpy as np

import paddle
import x2paddle
from x2paddle import torch2paddle
import x2paddle.torch2paddle as dist
from x2paddle.torch2paddle import create_tensor

'''
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
'''

class SmoothedValue(object):

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total],
                             dtype=paddle.float64,
                             device='cuda')
        paddle.distributed.barrier()
        paddle.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        data = list(self.deque)
        d = paddle.to_tensor(data)
        d_tensor = paddle.median(d)
        return d_tensor.item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    nparr = np.frombuffer(buffer, dtype=np.uint8)
    tensor = paddle.to_tensor(nparr, dtype=paddle.uint8).to('cuda')
    #storage = paddorch.ByteStorage.from_buffer(buffer)
    #tensor = torch2paddle.create_uint8_tensor(storage).to('cuda')
    local_size = paddle.to_tensor([tensor.numel()], device='cuda')
    size_list = [paddle.to_tensor([0], device='cuda') for _ in range(
        world_size)]
    paddle.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = []
    paddle.set_device("cuda")
    for _ in size_list:
        tensor_list.append(paddle.empty((max_size,), dtype=paddle.uint8))
    if local_size != max_size:
        padding = paddle.empty(shape=(max_size - local_size,), dtype=paddle
            .uint8)
        tensor = torch2paddle.concat((tensor, padding), dim=0)
    paddle.distributed.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(values, axis=0)
        paddle.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            #if isinstance(v, paddle.Tensor):
            if not isinstance(v, (float, int)):
                v = v.item()
            #assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if paddle.is_compiled_with_cuda():
            log_msg = self.delimiter.join([header, '[{0' + space_fmt +
                '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}',
                'data: {data}'])
            #'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header, '[{0' + space_fmt +
                '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}',
                'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if paddle.is_compiled_with_cuda():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)))
                else:
                    print(log_msg.format(i,
                                         len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
            total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command,
                                       cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = 'has uncommited changes' if diff else 'clean'
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f'sha: {sha}, status: {diff}, branch: {branch}'
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[create_tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = paddle.zeros(batch_shape, dtype=dtype).requires_grad_(False)
        mask = paddle.ones((b, h, w), dtype=paddle.bool).requires_grad_(False)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch2paddle.max(
            paddle.stack([img.shape[i]
                          for img in tensor_list]).to(paddle.float32)).to(paddle.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = paddle.nn.functional.pad(img, (0, padding[2], 0,
            padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = paddle.full_like(img[0], dtype=paddle.int32).requires_grad_(False)
        padded_mask = paddle.nn.functional.pad(m,
                                               (0, padding[2], 0, padding[1]),
                                               'constant', 1)
        padded_masks.append(padded_mask.to(paddle.bool))

    tensor = paddle.stack(padded_imgs)
    mask = paddle.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[create_tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    # if not paddle.distributed.is_available():
    #     return False
    # if not paddle.distributed.is_initialized():
    #     return False
    return False


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return paddle.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return paddle.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        x2paddle.torch2paddle.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch2paddle.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch2paddle.set_cuda_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.
        dist_url), flush=True)
    torch2paddle.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
    paddle.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@paddle.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [paddle.zeros([]).requires_grad_(False)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

'''
def interpolate(input, size=None, scale_factor=None, mode='nearest',
    align_corners=None):
    """Equivalent to nn.functional.interpolate, but with support for empty
    batch sizes.

    This will eventually be supported natively by PyTorch, and this class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return paddle.nn.functional.interpolate(input, size,
                scale_factor, mode, align_corners)
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size,
            scale_factor, mode, align_corners)
'''