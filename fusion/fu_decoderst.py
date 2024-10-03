"""by torchvision/detection
"""
import torch.nn as nn
import torch
from torch import Tensor
import math
import numpy as np


class build_positive_queries(nn.Module):
    def __init__(self,
                 anchors: Tensor,  # [3, 2]-> n, h, w
                 target: Tensor,
                 iou_t: float,
                 down_sample: int = 32,
                 ) -> None:
        super(build_positive_queries, self).__init__()
        self.anchors = anchors
        self.target = target
        self.iou_t = iou_t
        self.down_sample = down_sample

        self.encoding = nn.Linear

    def forward_positive(self, p: Tensor):
        nt = self.target.shape[0]
        positive = []
        ga_martix = torch.ones(6, device=self.target.device).long()
        anchors = self.anchors / self.down_sample
        ga_martix[2:] = torch.tensor(p.shape)[[3, 2, 3, 2]]
        na = anchors.shape[0]
        at = torch.arange(na).view(na, 1).repeat(1, nt)
        a, t, offsets = [], self.target * ga_martix, 0
        if nt:
            j = wh_iou(anchors, t[:, 4:6]) > self.iou_t
            a, t = at[j], t.repeat(na, 1, 1)[j]

        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # image_idx, anchor_idx, grid indices(y, x)
        positive.append((b, a, gj.clamp_(0, ga_martix[3] - 1), gi.clamp_(0, ga_martix[2] - 1)))
        positive_anchors = []
        for t in positive:
            w, h = self.anchors[t[1]]
            x, y = t[3], t[2]
            positive_anchors.append([x, y, w, h])

        return positive_anchors

    def forward(self, x: Tensor) -> Tensor:

        assert x.shape == 4

        p_1 = torch.randn((x.shape[0], 3, x.shape[2], x.shape[3]))
        positive_an = self.forward_positive(p_1)

        p_a = self.encoding(positive_an.shape[1], 256, bias=False)

        return p_a


'''ATSS by shenyudu Copy right(ATSS) 
https://github.com/sfzhang15/ATSS
'''


class PositiveLayer_ATSS(nn.Module):
    def __init__(self,
                 anchors: Tensor,
                 target: Tensor,
                 iou_m: str,
                 down_sample: int = 32,
                 top_k: int = 9,
                 ) -> None:
        super(PositiveLayer_ATSS, self).__init__()
        self.anchors = anchors
        self.target = target
        self.iou_m = iou_m
        # self.iou_t = iou_t
        self.down_sample = down_sample
        self.top_k = top_k
        self.MLP = nn.Conv2d

    def ATSS_assign(self, p_boxes: Tensor) -> Tensor:
        nt = self.target.shape[0]

        k_matrix_indices = torch.zeros((nt, self.top_k))
        for i in range(nt):
            gt_center = self.target[i, 2:4] # center point: [None, x, y, w, h]
            p_center = p_boxes[:, 2:4]  # center point: [None, x, y, w, h]
            dist_i = torch.sum((gt_center - p_center)**2, dim=-1) ** 0.5

            min_dist = torch.sort(dist_i, dim=-1).values
            min_index = min_dist.indices

            k_matrix_indices[i, :] = min_index[:9]

        iou_matrix_index = torch.zeros((nt, self.top_k))
        positive = []

        for i in range(nt):
            iou_matrix = bbox_iou(self.target[i, 2:6], p_boxes[[k_matrix_indices[i]], 2:6],
                                  x1y1x2y2=False,
                                  GIoU=True)
            iou_threshold = torch.mean(iou_matrix, dim=1) + torch.var(iou_matrix, dim=1)
            arg_index = torch.nonzero(iou_matrix > iou_threshold)
            positive.append(p_boxes[k_matrix_indices[arg_index], 2:6])

        return positive

    def forward(self, x):
        assert x.shape == 4

        p = self.MLP(x.shape[1], 3, 1, 1)
        positive = self.ATSS_assign(p)
        x1 = torch.cat((p[0], p[1], p[2]), dim=0)

        x1 = nn.Linear(4, 256, bias=False)

        return x1












@staticmethod
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


@staticmethod
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


@staticmethod
def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device).long()

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):

        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = anchors.shape[0]  # number of anchors
        # [3] -> [3, 1] -> [3, nt]
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))

            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T

        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)
        if c.shape[0]:
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


if __name__ == '__main__':
    '''test iou calculation
    '''
    proposal = torch.Tensor([10, 10, 30, 30])
    targets = torch.Tensor([[15, 17, 15, 17], [17, 19, 20, 20]])
    iou = bbox_iou(proposal, targets, x1y1x2y2=True)
    print(iou.shape)
    print(iou)
