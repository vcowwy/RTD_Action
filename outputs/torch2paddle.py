import paddle
import torch

# convert torch.pth to paddle.pdparams
chkpt_path = 'checkpoint_initial.pth'
paddle_chkpt_path = "checkpoint_initial.pdparams"

paddle_dict = {}
params_dict = {}
torch_checkpoint = torch.load(chkpt_path, map_location='cpu')

#start_epoch = torch_checkpoint['epoch']
paddle_dict['epoch'] = 0
pretrained_dict = torch_checkpoint['model']

# fc layer, weight needs to be tansposed
fc_names = ["transformer.encoder.layers.0.weight","transformer.encoder.layers.1.weight",
            "transformer.encoder.layers.2.weight","transformer.decoder.layers.0.linear1.weight",
            "transformer.decoder.layers.0.linear2.weight","transformer.decoder.layers.1.linear1.weight",
            "transformer.decoder.layers.1.linear2.weight","transformer.decoder.layers.2.linear1.weight",
            "transformer.decoder.layers.2.linear2.weight","transformer.decoder.layers.3.linear1.weight",
            "transformer.decoder.layers.3.linear2.weight","transformer.decoder.layers.4.linear1.weight",
            "transformer.decoder.layers.4.linear2.weight","transformer.decoder.layers.5.linear1.weight",
            "transformer.decoder.layers.5.linear2.weight","class_embed.weight",
            "bbox_embed.layers.0.weight", "bbox_embed.layers.1.weight",
            "bbox_embed.layers.2.weight","iou_embed.layers.0.weight",
            "iou_embed.layers.1.weight","iou_embed.layers.2.weight"]
print("Total FC Layers: {}".format(len(fc_names)))

count_fc = 0
count_in_proj = 0
for key in pretrained_dict:
    params = pretrained_dict[key].cpu().detach().numpy()
    flag = [item in key for item in fc_names]
    # FC Trans
    if any(flag):
        params = params.transpose()
        count_fc = count_fc + 1
        params_dict[key] = params
        continue
    # in_proj_weight 2 k,q,v weight
    if '.in_proj_weight' in key:
        count_in_proj = count_in_proj + 1
        step = params.shape[-1]
        q_name = key.replace('in','q').replace('_weight','.weight')
        k_name = key.replace('in', 'k').replace('_weight','.weight')
        v_name = key.replace('in', 'v').replace('_weight','.weight')
        params_dict[q_name] = params[:step].transpose()
        params_dict[k_name] = params[step:2*step].transpose()
        params_dict[v_name] = params[2*step:].transpose()
        continue
    # in_proj_bias 2 k,q,v bias
    if '.in_proj_bias' in key:
        assert step != 0
        q_name = key.replace('in', 'q').replace('_bias', '.bias')
        k_name = key.replace('in', 'k').replace('_bias', '.bias')
        v_name = key.replace('in', 'v').replace('_bias', '.bias')
        params_dict[q_name] = params[:step]
        params_dict[k_name] = params[step:2 * step]
        params_dict[v_name] = params[2 * step:]
        step = 0
        continue
    if '.out_proj.weight' in key:
        params_dict[key] = params.transpose()
        continue
    # normal
    params_dict[key] = params
print("{} FC Layer Params Transposed".format(count_fc))
print("{} in_proj Params Transformed".format(count_in_proj))

paddle_dict['model'] = params_dict
paddle.save(paddle_dict,paddle_chkpt_path)



