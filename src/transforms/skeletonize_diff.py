import torch
from torch import nn
import torch.nn.functional as F


class SkeletonizeDiff(nn.Module):
    def __init__(self, iterations=5):
        super().__init__()
        self.iters = iterations

    def soft_erode(self, volume):
        return -F.max_pool3d(-volume, kernel_size=3, stride=1, padding=1)

    def soft_dilate(self, volume):
        return F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)

    def soft_open(self, volume):
        return self.soft_dilate(self.soft_erode(volume))

    def forward(self, pred_prob, **batch):
        '''
        pred_prob: [B, D, H, W]
        '''

        pred_prob = pred_prob.unsqueeze(1)

        pred_prob1 = self.soft_open(pred_prob)
        skel = F.relu(pred_prob - pred_prob1)
        for _ in range(self.iters):
            pred_prob = self.soft_erode(pred_prob)
            pred_prob1 = self.soft_open(pred_prob)
            delta = F.relu(pred_prob - pred_prob1)
            skel = skel + F.relu(delta - skel*delta)

        return {'pred_skel': skel.squeeze(1)}


class SkeletonizeDiffFast(nn.Module):
    def __init__(self, iterations=5):
        super().__init__()
        self.iters = iterations

    def soft_erode(self, volume):
        p1 = -F.max_pool3d(-volume, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        p2 = -F.max_pool3d(-volume, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
        p3 = -F.max_pool3d(-volume, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, volume):
        return F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)

    def soft_open(self, volume):
        return self.soft_dilate(self.soft_erode(volume))

    def forward(self, pred_prob, **batch):
        '''
        pred_prob: [B, D, H, W]
        '''
        pred_prob = pred_prob.unsqueeze(1)
        pred_prob1 = self.soft_open(pred_prob)
        skel = F.relu(pred_prob - pred_prob1)
        for _ in range(self.iters):
            pred_prob = self.soft_erode(pred_prob)
            pred_prob1 = self.soft_open(pred_prob)
            delta = F.relu(pred_prob - pred_prob1)
            skel = skel + F.relu(delta - skel*delta)

        return {'pred_skel': skel.squeeze(1)}
