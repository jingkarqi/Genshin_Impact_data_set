import win32gui
import torch
import numpy as np
import cv2
from PIL import ImageGrab

from utils.augmentations import letterbox
from utils.general import (non_max_suppression, scale_coords)
from models.experimental import attempt_load

conf_thres = 0.65
iou_thres = 0.45
weights = 'best.pt'
color = (0, 255, 0)

wnd_name = '原神'

hwnd = win32gui.FindWindow(None, wnd_name)

# 获取窗口的位置
x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
x1 = x1 + 300
y1 = y1 + 300
#识别矩形区域位置
rect = (x, y, x1, y1)

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 载入模型
    model = attempt_load(weights, device=device)
    # 将模型的stride赋给stride变量 32
    stride = max(int(model.stride.max()), 32) # model stride
    while True:
        # 截取目标区域图像
        im = ImageGrab.grab(bbox=rect)
        # 将图像数据转换成np数组
        img0 = np.array(im)
        # 将图像缩放到指定尺寸
        img = letterbox(img0, stride=stride)[0]
        # 函数将一个内存不连续存储的数组转换为内存连续存储的数组,使得运行速度更快
        img = np.ascontiguousarray(img)
        # 把数组转换成张量，且二者共享内存
        img = torch.from_numpy(img).to(device)
        img = img.float()  # 确保使用全精度浮点数
        # 压缩数据维度
        img /= 255 # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        # 对tensor进行转置
        img = img.permute(0, 3, 1, 2)
        # Inference 模型推理
        pred = model(img, augment=False, visualize=False)[0]
        # NMS 非极大值抑制 即只输出概率最大的分类结果
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        # 处理预测识别结果
        for i, det in enumerate(pred): # per i
            if len(det):
                # Rescale boxes from img size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # 框选出检测结果
                for *xyxy, conf, cls in reversed(det):
                    cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 3)
        # 调用CV显示结果图
        b, g, r = cv2.split(img0)
        image_1 = cv2.merge([r, g, b])
        #缩放image_1为原来的0.8倍
        image_1 = cv2.resize(image_1, (int(image_1.shape[1] * 0.3), int(image_1.shape[0] * 0.3)))
        cv2.imshow("display", np.array(image_1))

        cv2.waitKey(1)


if __name__ == '__main__':
    run()

#释放所有占用
cv2.destroyAllWindows()