import torch
import os
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from tqdm import tqdm
import config
from dataset import LiverDataset3
from loss import Stair_loss, Stair_loss_Dynamic
from utils import calculate_cross_line
from nets.NetV3 import StairNet_DepthOut
import csv
import time

feature_size_h = config.feature_size_h
feature_size_w = config.feature_size_w
stride_h = config.stride_h
stride_w = config.stride_w
# Cuda or cpu
device = config.device
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def write2csv(path, row):
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)
        print("Write data successfully!")
        f.close()


def train_model(model, optimizer, lr_scheduler, dataload, val_loader, num_epochs):
    start_log = -1
    patience = 0
    folder_name, no_patience, is_dynamicloss = args.folder_name, args.patience_times, args.is_dynamicloss
    # initialize 2 weights of dynamic loss
    delta = 0.5
    lambda1 = 8
    lambda2 = 12
    train_log = [0]
    if os.path.exists('./logs/' + folder_name + '/log.csv'):
        os.remove('./logs/' + folder_name + '/log.csv')
    row = ['Time', 'epoch', 'loss', 'X_error', 'Y_error',
           'precision-B', 'recall-B', 'F1-score-B', 'IoU-B',
           'precision-R', 'recall-R', 'F1-score-R', 'IoU-R',
           'precision-M', 'recall-M', 'F1-score-M', 'IoU-M',
           'PA', 'MPA', 'MIOU']
    write2csv('./logs/' + folder_name + '/log.csv', row)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # For each epoch, Re-instantiate the dynamic loss
        if is_dynamicloss:
            criterion = Stair_loss_Dynamic(lambda1, lambda2)
        else:
            criterion = Stair_loss()
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        with tqdm(total=len(dataload)) as pbar:
            for img_x, img_d, blue, red, mask in dataload:
                step += 1
                inputs_rgb = img_x.to(device)
                inputs_depth = img_d.to(device)
                blues = (blue[0].float().to(device), blue[1].float().to(device), blue[2].to(device))
                reds = (red[0].float().to(device), red[1].float().to(device), red[2].to(device))
                masks = mask.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs_rgb)
                conf_loss, loc_loss, mask_loss, depth_loss = criterion(outputs, blues, reds, masks, inputs_depth)
                loss = conf_loss + loc_loss + depth_loss + mask_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_description(
                    "%d/%d,train_loss:%0.3f||conf_loss:%0.3f||loc_loss:%0.3f||seg_loss:%0.3f||depth_loss:%0.3f" %
                    (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(), conf_loss.item(), loc_loss.item(),
                     mask_loss.item(), depth_loss.item()))
                pbar.update(1)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        # Validation
        print('-' * 10, 'Starting Validation')
        model.eval()
        TP_b, FP_b, FN_b = 0, 0, 0
        TP_r, FP_r, FN_r = 0, 0, 0
        X_error, Y_error = 0.0, 0.0
        PA, MPA, MIOU = 0.0, 0.0, 0.0
        precision_b, recall_b, F1_score_b, iou_b = 0.0, 0.0, 0.0, 0.0
        precision_r, recall_r, F1_score_r, iou_r = 0.0, 0.0, 0.0, 0.0
        precision_m, recall_m, F1_score_m, iou_m = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_images_x, val_images_d, val_blue, val_red, val_masks in tqdm(val_loader):
                images_r = val_images_x.to(device)
                #images_d = val_images_d.to(device)
                outputs = model(images_r)
                tp_b, fp_b, fn_b, tp_r, fp_r, fn_r, x_error, y_error, pa, mpa, miou = val_eval(outputs, val_blue,
                                                                                               val_red, val_masks,
                                                                                               epoch)
                X_error = X_error + x_error
                Y_error = Y_error + y_error
                PA, MPA, MIOU = PA + pa, MPA + mpa, MIOU + miou
                TP_b, FP_b, FN_b = TP_b + tp_b, FP_b + fp_b, FN_b + fn_b
                TP_r, FP_r, FN_r = TP_r + tp_r, FP_r + fp_r, FN_r + fn_r
            if TP_b > 0:
                precision_b += TP_b / (TP_b + FP_b)
                recall_b += TP_b / (TP_b + FN_b)
                F1_score_b += 2 * TP_b / (2 * TP_b + FP_b + FN_b)
                iou_b += TP_b / (TP_b + FP_b + FN_b)
            if TP_r > 0:
                precision_r += TP_r / (TP_r + FP_r)
                recall_r += TP_r / (TP_r + FN_r)
                F1_score_r += 2 * TP_r / (2 * TP_r + FP_r + FN_r)
                iou_r += TP_r / (TP_r + FP_r + FN_r)
            if (TP_r + TP_b) > 0:
                precision_m += (TP_r + TP_b) / (TP_r + TP_b + FP_r + FP_b)
                recall_m += (TP_r + TP_b) / (TP_r + TP_b + FN_r + FN_b)
                F1_score_m += 2 * (TP_r + TP_b) / (2 * (TP_r + TP_b) + FP_r + FP_b + FN_r + FN_b)
                iou_m += (TP_r + TP_b) / (TP_r + TP_b + FP_r + FP_b + FN_r + FN_b)
            PA, MPA, MIOU = PA / len(val_loader), MPA / len(val_loader), MIOU / len(val_loader)
            adjustment = 0.0
            if X_error != 0 and Y_error != 0:
                if X_error > Y_error:
                    adjustment = (X_error - Y_error) / X_error
                    if (lambda1 + adjustment) > 20 - delta:
                        lambda1 = 20 - delta
                    else:
                        lambda1 = lambda1 + adjustment
                    if (lambda2 - adjustment) < delta:
                        lambda2 = delta
                    else:
                        lambda2 = lambda2 - adjustment
                else:
                    adjustment = (Y_error - X_error) / Y_error
                    if (lambda1 - adjustment) < delta:
                        lambda1 = delta
                    else:
                        lambda1 = lambda1 - adjustment
                    if (lambda2 + adjustment) > 20 - delta:
                        lambda2 = 20 - delta
                    else:
                        lambda2 = lambda2 + adjustment
            print('Adjustment value: %0.3f | Parameter1: %0.3f | Parameter2: %0.3f' % (adjustment, lambda1, lambda2))
            print('X_direction Error: %0.3f | Y_direction Error: %0.3f' % (X_error, Y_error))
            print('Blue line accuracy|precision: %0.3f recall: %0.3f F1-score: %0.3f IoU: %0.3f' % (
                precision_b, recall_b, F1_score_b, iou_b))
            print('Red line accuracy|precision: %0.3f recall: %0.3f F1-score: %0.3f IoU: %0.3f' % (
                precision_r, recall_r, F1_score_r, iou_r))
            print('Total Line accuracy|precision: %0.3f recall: %0.3f F1-score: %0.3f IoU: %0.3f' % (
                precision_m, recall_m, F1_score_m, iou_m))
            print('Semantic segmentation accuracy|PA: %0.3f MPA: %0.3f MIoU: %0.3f' % (PA, MPA, MIOU))
        if epoch > start_log:
            if (F1_score_m + MPA) > max(train_log):
                patience = 0
                torch.save(model.state_dict(), './logs/' + folder_name + '/best.pth')
                print('Best weights have been saved!')
            else:
                patience += 1
            train_log.append(F1_score_m + MPA)
            # Record training data
            torch.save(model.state_dict(), './logs/' + folder_name + '/last.pth')
            row = [time.asctime(), epoch, format(epoch_loss / step, '.4f'), format(X_error, '.4f'),
                   format(Y_error, '.4f'),
                   format(precision_b, '.4f'), format(recall_b, '.4f'), format(F1_score_b, '.4f'), format(iou_b, '.4f'),
                   format(precision_r, '.4f'), format(recall_r, '.4f'), format(F1_score_r, '.4f'), format(iou_r, '.4f'),
                   format(precision_m, '.4f'), format(recall_m, '.4f'), format(F1_score_m, '.4f'), format(iou_m, '.4f'),
                   format(PA, '.4f'), format(MPA, '.4f'), format(MIOU, '.4f')]
            write2csv('./logs/' + folder_name + '/log.csv', row)
        lr_scheduler.step()
        # End the training if the accuracy no longer improves
        if patience > no_patience:
            break
    # os.system("shutdown -s -t 60 ")
    return model


def train(args):
    batch_size, num_epoches, width_factor = args.batch_size, args.epochs, args.width_factor
    model = StairNet_DepthOut(width=width_factor).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.000001, lr=0.00025)
    # Apply dynamic learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_dataset = LiverDataset3("data/train", transform=train_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = LiverDataset3("data/val", transform=valid_transforms, is_train=False)
    val_dataloaders = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    train_model(model, optimizer, lr_scheduler, train_dataloaders, val_dataloaders, num_epoches)


def MPA(pred, label, classes=None):
    if classes is None:
        classes = [0, 1, 2]
    sum = [0, 0, 0]
    sum1 = [0, 0, 0]
    TP = [0.0, 0.0, 0.0]
    w = pred.shape[0]
    h = pred.shape[1]
    mpa, pa, iou = 0.0, 0.0, 0.0
    div, div1 = 0, 0
    for cls in range(len(classes)):
        for i in range(w):
            for j in range(h):
                if label[i, j] == classes[cls]:
                    sum[cls] += 1
                    if pred[i, j] == label[i, j]:
                        TP[cls] += 1
                if pred[i, j] == classes[cls]:
                    sum1[cls] += 1
        pa += TP[cls]
        if sum[cls] != 0:
            div += 1
            mpa += TP[cls] / sum[cls]
        if (sum1[cls] + sum[cls] - TP[cls]) != 0:
            div1 += 1
            iou += TP[cls] / (sum1[cls] + sum[cls] - TP[cls])

    return pa / w / h, mpa / div, iou / div1


# Calculate the accuracy
def val_eval(outputs, blues, reds, masks, epoch, start_idx=150):
    x_error = 0.0
    y_error = 0.0
    TP_b, FP_b, FN_b = 0, 0, 0
    TP_r, FP_r, FN_r = 0, 0, 0
    conf = args.conf
    fb1, fb2, fr1, fr2, f3, _ = outputs
    fb1 = fb1.cpu()
    fb2 = fb2.cpu()
    fr1 = fr1.cpu()
    fr2 = fr2.cpu()
    f3 = f3.cpu()
    # fb1 = torch.sigmoid(fb1)
    fb1 = torch.squeeze(fb1).detach().numpy()
    fb2 = torch.squeeze(fb2).detach().numpy()

    # fr1 = torch.sigmoid(fr1)
    fr1 = torch.squeeze(fr1).detach().numpy()
    fr2 = torch.squeeze(fr2).detach().numpy()

    clss_b = torch.squeeze(blues[0]).detach().numpy()
    labels_b = torch.squeeze(blues[1]).detach().numpy()
    clss_r = torch.squeeze(reds[0]).detach().numpy()
    labels_r = torch.squeeze(reds[1]).detach().numpy()
    for i in range(feature_size_h):
        for j in range(feature_size_w):
            # Calculate the total average error in the x and y directions
            if fb1[i][j] >= 0.5:
                x_error = x_error + abs(fb2[0][i][j] - labels_b[0][i][j]) + abs(fb2[2][i][j] - labels_b[2][i][j])
                y_error = y_error + abs(fb2[1][i][j] - labels_b[1][i][j]) + abs(fb2[3][i][j] - labels_b[3][i][j])
            if fr1[i][j] >= 0.5:
                x_error = x_error + abs(fr2[0][i][j] - labels_r[0][i][j]) + abs(fr2[2][i][j] - labels_r[2][i][j])
                y_error = y_error + abs(fr2[1][i][j] - labels_r[1][i][j]) + abs(fr2[3][i][j] - labels_r[3][i][j])
            # Calculate the errors of the blue and red lines separately
            line_p_b = [fb2[0][i][j], fb2[1][i][j], fb2[2][i][j], fb2[3][i][j]]
            line_g_b = [labels_b[0][i][j], labels_b[1][i][j], labels_b[2][i][j], labels_b[3][i][j]]
            # If contains blue lines
            if fb1[i][j] >= 0.5:
                if clss_b[i][j] >= 0.5:
                    if calculate_cross_line(line_p_b, line_g_b) >= conf:
                        TP_b += 1
                    else:
                        FP_b += 1
                else:
                    FP_b += 1
            else:
                if clss_b[i][j] >= 0.5:
                    FN_b += 1

            line_p_r = [fr2[0][i][j], fr2[1][i][j], fr2[2][i][j], fr2[3][i][j]]
            line_g_r = [labels_r[0][i][j], labels_r[1][i][j], labels_r[2][i][j], labels_r[3][i][j]]
            # If contains red lines
            if fr1[i][j] >= 0.5:
                if clss_r[i][j] >= 0.5:
                    if calculate_cross_line(line_p_r, line_g_r) >= conf:
                        TP_r += 1
                    else:
                        FP_r += 1
                else:
                    FP_r += 1
            else:
                if clss_r[i][j] >= 0.5:
                    FN_r += 1
    # Accuracy for segmentation
    if epoch >= start_idx and epoch % 2 == 0:
        _, predicted = torch.max(f3.data, 1)
        pred = torch.squeeze(predicted).numpy()
        _, masks = torch.max(masks, 1)
        masks = torch.squeeze(masks).numpy()
        pa, mpa, iou = MPA(pred, masks)
    else:
        pa, mpa, iou = 0.0, 0.0, 0.0
    return TP_b, FP_b, FN_b, TP_r, FP_r, FN_r, x_error, y_error, pa, mpa, iou


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=300)
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--width_factor", type=float, help="for scaling of models", default=0.75)
    parse.add_argument("--folder_name", type=str, help="the folder of model weights",
                       default="StairNet_DepthOut_0.75")
    parse.add_argument('--patience_times', type=int, help="break training after patience_times", default=60)
    parse.add_argument('--is_dynamicloss', type=bool, help="apply dynamic loss or not", default=True)
    parse.add_argument('--conf', type=float, default=0.5, help='conf for validation')
    args = parse.parse_args()
    train(args)

