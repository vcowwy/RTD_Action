import paddle
import numpy as np


class DistributedSampler(paddle.io.DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)


def LongTensor(x):
    if isinstance(x, int):
        return paddle.to_tensor([x], dtype="int64")
    if isinstance(x, list):
        x = paddle.to_tensor(x, dtype="int64")
    return x


class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False
        return_list = True
        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler


class Linear(paddle.nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        uniform_bound = np.sqrt(1.0/in_features)
        weight_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
        if not bias:
            bias_attr = False
        else:
            bias_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
                #paddle.ParamAttr(name="bias")
        super(Linear, self).__init__(in_features,
                                     out_features,
                                     weight_attr=weight_attr,
                                     bias_attr=bias_attr)


class Conv1d(paddle.nn.Conv1D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        uniform_bound = np.sqrt(1.0 / in_channels)
        weight_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
        if not bias:
            bias_attr = False
        else:
            bias_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
        #bias_attr = None
        #if not bias:
        #    bias_attr = False
        #else:
        #    bias_attr =paddle.nn.initializer.MSRAInitializer() # fluid.initializer.ConstantInitializer(value=0)

        super(Conv1d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     padding_mode=padding_mode,
                                     weight_attr=weight_attr,
                                     bias_attr=bias_attr,
                                     data_format='NCL')


class Conv2d(paddle.nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        uniform_bound = np.sqrt(1.0 / in_channels)
        weight_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
        if not bias:
            bias_attr = False
        else:
            bias_attr = paddle.nn.initializer.Uniform(-uniform_bound, uniform_bound)
        # bias_attr = None
        # if not bias:
        #    bias_attr = False
        # else:
        #    bias_attr =paddle.nn.initializer.MSRAInitializer() # fluid.initializer.ConstantInitializer(value=0)

        super(Conv2d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     padding_mode=padding_mode,
                                     weight_attr=weight_attr,
                                     bias_attr=bias_attr,
                                     data_format='NCHW')