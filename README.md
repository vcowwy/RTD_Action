#论文《Relaxed Transformer Decoders for Direct Action Proposal Generation》


## 遇到的问题
### 训练对齐

engine.py文件中train_one_epoch的losses.backward()报错如下：

![img_1](/img/img_1.png)

前向和评估指标都与pytorch实现对齐了，但pytorch实现的losses.backward()不会出现这个报错信息。

### 分析报错原因

⭐ 根据它的报错提示1

You use some output variables of the previous batch as the inputs of the current batch.

engine.py文件中每个batch的代码如下：

![img_2](/img/img_2.png)

由上，不存在这个batch使用了上个batch数据的情况。

⭐ 根据它的报错提示2

1.应该不存在需要计算两次梯度的子图。

2.设置losses.backward()的参数retain_graph为True，训练三四个epoch之后就会导致显卡内存一直增加到out of memory。

⭐ 调试结果

如果把网络中transformer所有decoder的参数stop_gradient设置为True，loss可以正常backward。

## 最新进展

### 网络初始化

pytorch实现的网络初始化后，将网络初始化信息保存在outputs/checkpoint_initial.pth中，使用torch2paddle.py将其转换为checkpoint_initial.pdparams


### 训练

在paddle实现中使用命令：

`python main.py --window_size 100 --batch_size 32 --stage 1 --num_queries 32 --point_prob_normalize --resume outputs/checkpoint_initial.pdparams --dropout 0`

训练4个epoch后，虽然会因out of memory而报错，但网络信息都保存在outputs/checkpoint.pdparams文件里了，因此使用命令：

`python main.py --window_size 100 --batch_size 32 --stage 1 --num_queries 32 --point_prob_normalize --resume outputs/checkpoint.pdparams --dropout 0`

训练接下来的epoch，由于训练50个epoch时间过长，因此只训练10个epoch，将结果保存在log/paddle_log.txt


在pytorch实现中使用命令：

`python main.py --window_size 100 --batch_size 32 --stage 1 --num_queries 32 --point_prob_normalize --dropout 0`

训练10个epoch后将结果保存在log/torch_log.txt中

### 对比结果

训练10个epoch后:

pytorch实现的AR@50为18.70

![img_4](/img/img_4.png)



paddle实现的AR@50为22.77

![img_3](/img/img_3.png)
