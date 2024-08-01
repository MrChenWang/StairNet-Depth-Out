import argparse
import shutil
import glob
from pathlib import Path
from nets.NetV3 import StairNet_DepthOut
import torch
import os
import config
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import time
from utils import Draw_results

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class LoadImages:  # for inference
    def __init__(self, RGB_path, Depth_path, img_size=512):
        # get RGB images
        RGB_path = str(Path(RGB_path))
        RGB_files = []
        if os.path.isdir(RGB_path):
            RGB_files = sorted(glob.glob(os.path.join(RGB_path, '*.*')))
        elif os.path.isfile(RGB_path):
            RGB_files = [RGB_path]
        RGB_images = [x for x in RGB_files if os.path.splitext(x)[-1].lower() in img_formats]
        # get Depth images
        Depth_path = str(Path(Depth_path))
        Depth_files = []
        if os.path.isdir(Depth_path):
            Depth_files = sorted(glob.glob(os.path.join(Depth_path, '*.*')))
        elif os.path.isfile(Depth_path):
            Depth_files = [Depth_path]
        Depth_images = [x for x in Depth_files if os.path.splitext(x)[-1].lower() in img_formats]

        self.img_size = img_size

        self.RGB_files = RGB_images
        self.nRGB = len(RGB_images)  # number of files
        self.Depth_files = Depth_images
        self.nDepth = len(Depth_images)

        assert self.nRGB > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s' % \
                              (RGB_path, img_formats)

        assert self.nRGB == self.nDepth, 'The numbers of RGB and Depth images are different!'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nRGB:
            raise StopIteration
        RGB_path = self.RGB_files[self.count]
        Depth_path = self.Depth_files[self.count]

        self.count += 1
        img0 = cv2.imread(RGB_path)
        img0_d = cv2.imread(Depth_path, cv2.IMREAD_GRAYSCALE)
        assert img0 is not None, 'Image Not Found ' + RGB_path
        assert img0_d is not None, 'Image Not Found ' + Depth_path
        print('image %g/%g %s: ' % (self.count, self.nRGB, RGB_path), end='')

        # padded and resize
        padh = (img0.shape[1] - img0.shape[0]) // 2

        img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.ascontiguousarray(img)

        img_d = np.pad(img0_d, ((padh, padh), (0, 0)), 'constant', constant_values=0)
        img_d = cv2.resize(img_d, (self.img_size, self.img_size))
        img_d = np.ascontiguousarray(img_d)

        return RGB_path, img, img_d, img0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nRGB  # number of files


def detect():
    width_factor, inPath, out, source, weights, view_img, conf_thres, imgsz = \
        opt.width_factor, opt.input, opt.output, opt.source, opt.weights, opt.view_img, opt.conf_thres, opt.img_size
    # initialization
    device = config.device
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = StairNet_DepthOut(width=width_factor).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    if half:
        model.half()  # to FP16
    # Set Dataloader
    # TODO online inference for webcam and depth camera
    if source == 0:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams(source, img_size=imgsz)
    elif source == 1:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams_depthcamera(source, img_size=imgsz)
    elif source == 2:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(inPath, inPath.replace('images', 'depthes'), img_size=imgsz)
    else:
        print("Please choose the correct index for source!")

    for path, img, img_d, img0 in dataset:
        # Process RGB image
        input = (img / 255.0 - 0.5) / 0.5
        x = torch.Tensor(input)
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=0)
        x = x.to(device)
        x = x.half() if half else x.float()

        # torch.cuda.synchronize()
        start = time.time()
        y = model(x)
        # torch.cuda.synchronize()
        endt1 = time.time()
        print("CNN inference time:" + format(endt1 - start, '.3f') + 's')

        fb1, fb2, fr1, fr2, y3, _ = y
        fb1, fb2, fr1, fr2 = fb1.cpu(), fb2.cpu(), fr1.cpu(), fr2.cpu()
        fb1 = torch.squeeze(fb1).detach().numpy()
        fb2 = torch.squeeze(fb2).detach().numpy()

        fr1 = torch.squeeze(fr1).detach().numpy()
        fr2 = torch.squeeze(fr2).detach().numpy()
        img_result = Draw_results(img0, fb1, fb2, fr1, fr2, y3, conf=conf_thres)
        if view_img:
            cv2.imwrite(os.path.join(out, os.path.basename(path)), img_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--width_factor", type=float, help="for scaling of models", default=1.0)
    parser.add_argument('--weights', type=str, default='logs/StairNet_DepthOut_1.0/best.pth', help='model.pth path')
    parser.add_argument('--source', type=int, default=2, help='source')  # 0 for webcam, 1 for realsense, 2 for files
    parser.add_argument('--input', type=str, default='data/val/images', help='output folder')  # input folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--view-img', type=bool, default=True, help='store results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
