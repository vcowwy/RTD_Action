#目前问题

## loss.backward()报错

1. You use some output variables of the previous batch as the inputs of the current batch. Please try to call "stop_gradient = True" or "detach()" for these variables.
2. You calculate backward twice for the same subgraph without setting retain_graph=True. Please set retain_graph=True in the first backward call.

[Hint: Expected ins_.empty() && outs_.empty() != true, but received ins_.empty() && outs_.empty():1 == true:1.] (at /paddle/paddle/fluid/imperative/op_base.h:147)
(NotFound) Inputs and outputs of assign do not exist. This may be because:


### 解决方法1

设置retain_graph=True，loss.backward(retain_graph=True)

不可行，会导致显卡内存随着训练一直增加直到out of memory。
。
