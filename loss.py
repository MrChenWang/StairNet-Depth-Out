import torch
import torch.nn as nn
import config

device = config.device
feature_size_h = config.feature_size_h
feature_size_w = config.feature_size_w


class Stair_loss(nn.Module):
    def __init__(self):
        super(Stair_loss, self).__init__()
        self.conf_c = nn.BCEWithLogitsLoss()
        self.loc_c = nn.MSELoss(reduction='none')

    def forward(self, outputs, masks, labels):
        y1, y2 = outputs
        n_batch = labels.size()[0]  # batch size
        conf_loss = self.conf_c(y1, masks)

        loc_vector = self.loc_c(y2, labels)

        # Data slicing for x and y location losses
        loc_vector_x = loc_vector[:, 0:8:2, :, :]
        loc_vector_y = loc_vector[:, 1:8:2, :, :]
        # print(loc_vector_x.size(), loc_vector_y.size())
        loc_sum_x = torch.sum(loc_vector_x, dim=1, keepdim=True)
        loc_sum_y = torch.sum(loc_vector_y, dim=1, keepdim=True)
        # print(loc_sum_x.size(), loc_sum_y.size())

        # Multiply each element one by one
        loc_loss_x = torch.mul(masks, loc_sum_x)
        loc_loss_y = torch.mul(masks, loc_sum_y)

        # print(loc_loss_x.size(), loc_loss_y.size())
        loc_loss_x = torch.sum(loc_loss_x)
        loc_loss_y = torch.sum(loc_loss_y)
        # Give weights of 1:4
        loc_loss = 4 * loc_loss_x + 16 * loc_loss_y
        loc_loss = loc_loss / (n_batch * feature_size_h * feature_size_w)

        print("conf_loss:%0.3f||loc_loss:%0.3f" % (conf_loss.item(), loc_loss.item()))
        return conf_loss + loc_loss


class Stair_loss_Dynamic(nn.Module):
    def __init__(self, lambda1=10, lambda2=10):
        super(Stair_loss_Dynamic, self).__init__()
        self.conf_b = nn.MSELoss(reduction='none')
        self.conf_r = nn.MSELoss(reduction='none')
        self.loc_b = nn.MSELoss(reduction='none')
        self.loc_r = nn.MSELoss(reduction='none')
        self.mask_c = nn.BCEWithLogitsLoss()
        self.depth_c = nn.MSELoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, outputs, blues, reds, masks, depthes):
        fb1, fb2, fr1, fr2, f3, f4 = outputs
        n_batch = masks.size()[0]  # batch size

        conf_vector_b = self.conf_b(fb1, blues[0])
        # Get the positive loss
        conf_loss_b_P = torch.mul(blues[2], conf_vector_b)
        conf_loss_b_P = torch.sum(conf_loss_b_P) / (n_batch * feature_size_h * feature_size_w)
        # Get the negative loss
        conf_loss_b_N = torch.mul(torch.ones(n_batch, 1, feature_size_h, feature_size_w).to(device) - blues[2],
                                  conf_vector_b)
        conf_loss_b_N = torch.sum(conf_loss_b_N) / (n_batch * feature_size_h * feature_size_w)
        conf_loss_b = 15 * conf_loss_b_P + 5 * conf_loss_b_N  # Give weights of 3:1

        conf_vector_r = self.conf_r(fr1, reds[0])
        conf_loss_r_P = torch.mul(reds[2], conf_vector_r)
        conf_loss_r_P = torch.sum(conf_loss_r_P) / (n_batch * feature_size_h * feature_size_w)
        conf_loss_r_N = torch.mul(torch.ones(n_batch, 1, feature_size_h, feature_size_w).to(device) - reds[2],
                                  conf_vector_r)
        conf_loss_r_N = torch.sum(conf_loss_r_N) / (n_batch * feature_size_h * feature_size_w)
        conf_loss_r = 15 * conf_loss_r_P + 5 * conf_loss_r_N
        # print(y1.size(), masks.size())

        loc_vector_b = self.loc_b(fb2, blues[1])

        # Data slicing for x and y location losses
        loc_vector_x_b = loc_vector_b[:, 0:4:2, :, :]
        loc_vector_y_b = loc_vector_b[:, 1:4:2, :, :]
        # print(loc_vector_x.size(), loc_vector_y.size())
        loc_sum_x_b = torch.sum(loc_vector_x_b, dim=1, keepdim=True)
        loc_sum_y_b = torch.sum(loc_vector_y_b, dim=1, keepdim=True)
        # print(loc_sum_x.size(), loc_sum_y.size())
        # Multiply each element one by one
        loc_loss_x_b = torch.mul(blues[2], loc_sum_x_b)
        loc_loss_y_b = torch.mul(blues[2], loc_sum_y_b)

        # print(loc_loss_x.size(), loc_loss_y.size())
        loc_loss_x_b = torch.sum(loc_loss_x_b)
        loc_loss_y_b = torch.sum(loc_loss_y_b)
        # Apply dynamic weights
        loc_loss_b = self.lambda1 * loc_loss_x_b + self.lambda2 * loc_loss_y_b
        loc_loss_b = loc_loss_b / (n_batch * feature_size_h * feature_size_w)

        loc_vector_r = self.loc_r(fr2, reds[1])

        # Data slicing for x and y location losses
        loc_vector_x_r = loc_vector_r[:, 0:4:2, :, :]
        loc_vector_y_r = loc_vector_r[:, 1:4:2, :, :]
        # print(loc_vector_x.size(), loc_vector_y.size())
        loc_sum_x_r = torch.sum(loc_vector_x_r, dim=1, keepdim=True)
        loc_sum_y_r = torch.sum(loc_vector_y_r, dim=1, keepdim=True)
        # print(loc_sum_x.size(), loc_sum_y.size())
        # Multiply each element one by one
        loc_loss_x_r = torch.mul(reds[2], loc_sum_x_r)
        loc_loss_y_r = torch.mul(reds[2], loc_sum_y_r)

        # print(loc_loss_x.size(), loc_loss_y.size())
        loc_loss_x_r = torch.sum(loc_loss_x_r)
        loc_loss_y_r = torch.sum(loc_loss_y_r)
        # Apply dynamic weights
        loc_loss_r = self.lambda1 * loc_loss_x_r + self.lambda2 * loc_loss_y_r
        loc_loss_r = loc_loss_r / (n_batch * feature_size_h * feature_size_w)

        mask_loss = self.mask_c(f3, masks)
        depth_loss = 10 * self.depth_c(f4, depthes)
        return conf_loss_b + conf_loss_r, loc_loss_b + loc_loss_r, mask_loss, depth_loss



