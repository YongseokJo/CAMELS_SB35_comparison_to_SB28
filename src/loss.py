import torch
import torch.nn as nn

class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(LogMSELoss, self).__init__()
        self.eps = eps  # to prevent log(0)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mean = torch.mean((pred-target)**2,dim=0)
        lmse = torch.sum(torch.log(mean))
        return lmse
        """
        pred_log = torch.log(pred)
        target_log = torch.log(target)
        return torch.mean((pred_log - target_log) ** 2)
        """

