import torch
from torch import nn
from torch.autograd import Variable


def get_rmse_log(model, feature, label, use_gpu):
    model.eval()
    mse_loss = nn.MSELoss()
    if use_gpu:
        feature = feature.cuda()
        label = label.cuda()
    feature = Variable(feature, volatile=True)
    label = Variable(label, volatile=True)
    pred = model(feature)
    clipped_pred = torch.clamp(pred, 1, float('inf'))
    rmse = torch.sqrt(mse_loss(clipped_pred.log(), label.log()))
    return rmse.data[0]
