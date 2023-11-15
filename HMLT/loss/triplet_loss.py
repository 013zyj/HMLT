import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, method, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    if method is 'enc':
        sorted_mat_distance, _ = torch.sort(dist_mat[is_pos].contiguous().view(N, -1), dim=1, descending=True)
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        mask = sorted_mat_distance > 0.05
        post_dist = torch.masked_select(sorted_mat_distance, mask)  # 形成一维张量
        post_dist, _ = torch.sort(post_dist)
        post_dist = torch.mean(post_dist)
#         print(post_dist)
        # print(dist_mat[is_pos].shape)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)
        return dist_ap, dist_an, post_dist
    if method is 'cos':
        sorted_mat_distance, _ = torch.sort(dist_mat + (9999999.) * (~is_pos), dim=1, descending=False)
        hard_p = sorted_mat_distance[:, 0]
        sorted_mat_distance, _ = torch.sort(dist_mat + (-9999999.) * (is_pos), dim=1, descending=True)
        hard_n = sorted_mat_distance[:, 0]
        # hard_p, relative_p_inds = torch.min(
        #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # hard_n, relative_n_inds = torch.max(
        #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        # hard_p = hard_p.squeeze(1)
        # hard_n = hard_n.squeeze(1)
        return hard_p, hard_n




class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=0.3, hard_factor=0.0, beta=1):
        self.margin = margin
        self.hard_factor = hard_factor
        self.beta = beta
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def hard_sigmoid_post(self, x, c=12):

        """
        假设x=6
        post_dist: c=12
        post_neighbor: c=3
        """
        if x < 0:
            return 0
        elif x > 2 * c:
            return 1
        else:
            return x / (2 * c) + 0.002  # 6/(2*12)+0.02=0.45

    def negative_sample_distribute(self, x, d=12):

        """
        假设x=6
        post_dist: c=12
        post_neighbor: c=3
        """
        if x > 2*d:
            return 0
        else:
            return x / (2 * d) + 0.002  # 6/(2*12)+0.02=0.45
    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        enc_dist = euclidean_dist(global_feat, global_feat)
        enc_ap, enc_an, post_dist = hard_example_mining(enc_dist, labels, 'enc')
        cos_dist = cosine_dist(global_feat, global_feat)
        cos_ap, cos_an = hard_example_mining(cos_dist, labels, 'cos')
        enc_ap *= (1.0 + self.hard_factor)
        enc_an *= (1.0 - self.hard_factor)

        y = enc_an.new().resize_as_(enc_an).fill_(1)
        y1 =-torch.ones_like(cos_ap)
        if self.margin is not None:

            enc_loss = self.ranking_loss(enc_an, enc_ap, y)
            cos_loss = self.ranking_loss(cos_an, cos_ap, y1)
            euc_pos_loss = self.hard_sigmoid_post(post_dist, c=7)
            negative_loss = self.negative_sample_distribute(post_dist)
            loss = enc_loss + cos_loss + self.beta*euc_pos_loss + self.beta*negative_loss
            # loss = enc_loss + cos_loss
        else:
            enc_loss = self.ranking_loss(enc_an - enc_ap, y)
            cos_loss = self.ranking_loss(cos_an - cos_ap, y1)
            euc_pos_loss = self.hard_sigmoid_post(post_dist, c=7)
            negative_loss = self.negative_sample_distribute(post_dist)
            loss = enc_loss + cos_loss + self.beta * euc_pos_loss + self.beta * negative_loss
            # loss = enc_loss + cos_loss 
        return loss, enc_ap, enc_an
#         return loss