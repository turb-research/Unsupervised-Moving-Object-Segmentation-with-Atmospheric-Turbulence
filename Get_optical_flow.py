import numpy as np
import cv2
import torch
import torch.nn.functional as F

# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision.transforms as T
# import kornia.geometry.transform as kg

from glob import glob
# import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as io
from PIL import Image
from tqdm import tqdm
import cv2
import einops
# from skimage import restoration, img_as_float
# from scipy.signal import gaussian
from torchvision.transforms import functional as TF

import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
# import matplotlib.pyplot as plt


# Load Images
def load_images(path, ResizeFactor=2):
    imgList = sorted(glob(path))

    oh, ow = io.imread(imgList[0]).shape[:2]
    h, w = ((oh // (8*ResizeFactor)) * 8, (ow // (8*ResizeFactor)) * 8) if oh > 0 and ow > 0 else (0, 0)
    resizeRatio, resizeRatioY = (h/oh, w/ow) if oh > 0 and ow > 0 else (0, 0)

    assert h > 0 and w > 0, "Invalid original dimensions!"

    imgTensor = torch.stack([T.Compose([T.Resize((h, w)), T.ToTensor()])(Image.open(img)) for img in imgList]) #torch.Size([time, ch, h, w])
    return imgTensor

# Load Images
def load_images_gray(path, ResizeFactor=2):
    imgList = sorted(glob(path))

    oh, ow = io.imread(imgList[0]).shape[:2]
    h, w = ((oh // (8*ResizeFactor)) * 8, (ow // (8*ResizeFactor)) * 8) if oh > 0 and ow > 0 else (0, 0)
    resizeRatio, resizeRatioY = (h/oh, w/ow) if oh > 0 and ow > 0 else (0, 0)

    assert h > 0 and w > 0, "Invalid original dimensions!"

    imgTensor = torch.stack([T.Compose([T.Resize((h, w)), T.Grayscale(), T.ToTensor()])(Image.open(img)) for img in imgList]) #torch.Size([time, ch, h, w])
    return imgTensor


# Function for Global Video Segmentation
def otsu_threshold_from_histogram(hist):
    total = np.sum(hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(256), hist)
    for i in range(256):
        wB += hist[i]
        wF = total - wB
        if wB == 0 or wF == 0:
            continue
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            level = i
            maximum = between
    return level


# Get Optical Flow Mask from image Sequences
def getFlowMask(imgTensor, ResizeFactor=2, deviceId = "cuda", n = 5):

    """
    imgList: List of image paths can be extracted from glob
    ResizeFactor: Resize factor for the image. Default is 2 times downsampled
    deviceId: Device ID for the GPU. Default is "cuda" so that it doesn't use the main GPU
    n = 5: Number of frames to consider for the optical flow. Default is 5
    """
    
    N, C, H, W = imgTensor.shape
    resized = False
    if max(imgTensor.shape) > 1000:
        resized = True
        ResizeFactor = 2
        imgTensor = F.interpolate(imgTensor, size=((H//ResizeFactor//8)*8, (W//ResizeFactor//8)*8), mode='bilinear', align_corners=False)
    
    # Resize the images by ResizeFactor to Downsample
    

    device = deviceId if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()

    batch = 1
    SegMaskList = []
    imgTensor = imgTensor.to(device)

    for idx in tqdm(range(0, len(imgTensor))):
        HSVStack = []

        # Loop through max(i-n, 0) to min(len(dataset), i+n)
        for diff in range(-n, n + 1):
            if diff == 0 or idx + diff < 0 or idx + diff >= len(imgTensor):
                continue
            
            imgTensor1 = imgTensor[idx: idx + batch]
            imgTensor2 = imgTensor[idx + diff: idx + batch + diff]

            with torch.no_grad():
                out = model(imgTensor1, imgTensor2)[-1][0]
                RGB = flow_to_image(out)
                RGB = einops.rearrange(RGB, 'c h w -> h w c').cpu().numpy()
                HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
                HSVStack.append(HSV)
            
        HSVStack = np.array(HSVStack)
        HSVMeanList = [np.mean(HSVStack[max(0, i - n):min(len(HSVStack), i + n + 1)], axis=0)[:, :, 1].astype(np.uint8) for i in range(len(HSVStack))]
        
        # Calculate perfectness of the mask
        minDistance = 255
        perfectMean = HSVMeanList[0].copy()
        HM_dis_list = []
        
        for HM in HSVMeanList:
            # Normalize the mask
            HM = ((HM - np.min(HM)) / (np.max(HM) - np.min(HM)) * 255).astype(np.uint8)
            distance = 128 - np.mean(abs(128 - HM.astype(int)))
            HM_dis_list.append(distance)
            if distance < minDistance:
                minDistance = distance
                perfectMean = HM

        # Save the perfect mean
        _, thresh = cv2.threshold(np.uint8(perfectMean), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        SegMaskList.append(thresh)
        
    return SegMaskList


def getFlowMaskGlobal(imgTensor, ResizeFactor=2, deviceId = "cuda", n = 5, ThMagnify=1.5):

    """
    imgList: List of image paths can be extracted from glob
    ResizeFactor: Resize factor for the image. Default is 2 times downsampled
    deviceId: Device ID for the GPU. Default is "cuda" so that it doesn't use the main GPU
    n = 5: Number of frames to consider for the optical flow. Default is 5
    """
    N, C, H, W = imgTensor.shape
    resized = False
    if max(imgTensor.shape) > 1000:
        resized = True
        ResizeFactor = 2
        imgTensor = F.interpolate(imgTensor, size=((H//ResizeFactor//8)*8, (W//ResizeFactor//8)*8), mode='bilinear', align_corners=False)

    # Resize the images by ResizeFactor to Downsample


    device = deviceId if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()

    batch = 1
    SegMaskList = []
    perfectMeanList = []
    imgTensor = imgTensor.to(device)

    combined_histogram = np.zeros(256)  # Step 1

    for idx in tqdm(range(0, len(imgTensor))):
        HSVStack = []

        # Loop through max(i-n, 0) to min(len(dataset), i+n)
        for diff in range(-n, n + 1):
            if diff == 0 or idx + diff < 0 or idx + diff >= len(imgTensor):
                continue
            
            imgTensor1 = imgTensor[idx: idx + batch]
            imgTensor2 = imgTensor[idx + diff: idx + batch + diff]

            with torch.no_grad():
                out = model(imgTensor1, imgTensor2)[-1][0]
                # RGB = flow_to_image(out)
                # RGB = einops.rearrange(RGB, 'c h w -> h w c').cpu().numpy()
                # HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)                  #(480, 992, 3)

                # Measure the magnitude of the flow
                HSV = torch.sqrt(torch.sum(out ** 2, dim=0, keepdim=True)).cpu().numpy()[0]
                HSV = np.stack([HSV, HSV, HSV], axis=2)

                HSVStack.append(HSV)
            
        HSVStack = np.array(HSVStack)
        HSVMeanList = [np.mean(HSVStack[max(0, i - n):min(len(HSVStack), i + n + 1)], axis=0)[:, :, 1].astype(np.uint8) for i in range(len(HSVStack))]
        
        # Calculate perfectness of the mask
        minDistance = np.max(HSVMeanList)                # Initial Assign, will be updated
        perfectMean = HSVMeanList[0].copy()
        HM_dis_list = []
        
        for HM in HSVMeanList:
            # Normalize the mask
            # HM = ((HM - np.min(HM)) / (np.max(HM) - np.min(HM)) * 255).astype(np.uint8)
            distance = np.max(HM)/2 - np.mean(abs(np.max(HM)/2 - HM))
            HM_dis_list.append(distance)
            if distance < minDistance:
                minDistance = distance
                perfectMean = HM
        
        hist, _ = np.histogram(perfectMean, bins=np.arange(256 + 1))
        combined_histogram += hist

        perfectMeanList.append(perfectMean)

        # Now Normalize it from 0 to 255
        # perfectMean = ((perfectMean - np.min(perfectMean)) / (np.max(perfectMean) - np.min(perfectMean)) * 255).astype(np.uint8)
        # Save the perfect mean
        _, thresh = cv2.threshold(np.uint8(perfectMean), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        SegMaskList.append(thresh)


        
    # Calculate the global Otsu's threshold from the combined_histogram
    global_otsu_thresh = otsu_threshold_from_histogram(combined_histogram)

    # Step 4: Apply global Otsu's threshold to each frame
    GlobalSegMaskList = []
    for perfectMean in perfectMeanList:
        _, thresh = cv2.threshold(np.uint8(perfectMean), global_otsu_thresh/ThMagnify, 255, cv2.THRESH_BINARY)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        GlobalSegMaskList.append(thresh)

    return GlobalSegMaskList


###############For ptlflow based optical flow calcculation#####################
def getFlowMask_other(imgTensor, of_model, train_datset, ResizeFactor=2, deviceId = "cuda", n = 5):

    """
    imgList: List of image paths can be extracted from glob
    ResizeFactor: Resize factor for the image. Default is 2 times downsampled
    deviceId: Device ID for the GPU. Default is "cuda" so that it doesn't use the main GPU
    n = 5: Number of frames to consider for the optical flow. Default is 5
    """
    
    N, C, H, W = imgTensor.shape
    resized = False
    if max(imgTensor.shape) > 1000:
        resized = True
        ResizeFactor = 2
        imgTensor = F.interpolate(imgTensor, size=((H//ResizeFactor//8)*8, (W//ResizeFactor//8)*8), mode='bilinear', align_corners=False)
    
    # Resize the images by ResizeFactor to Downsample
    

    device = deviceId if torch.cuda.is_available() else "cpu"

    # model = ptlflow.get_model("pwcnet",pretrained_ckpt='things')
    model = ptlflow.get_model(of_model,pretrained_ckpt=train_datset)
    model = model.to(device)


    batch = 1
    SegMaskList = []
    imgTensor = imgTensor.to(device)

    
    for idx in tqdm(range(0, len(imgTensor))):
        HSVStack = []

        # Loop through max(i-n, 0) to min(len(dataset), i+n)
        for diff in range(-n, n + 1):
            if diff == 0 or idx + diff < 0 or idx + diff >= len(imgTensor):
                continue
            
            imgTensor1 = imgTensor[idx: idx + batch]
            imgTensor2 = imgTensor[idx + diff: idx + batch + diff]

            images = [imgTensor1.squeeze(0).permute(1,2,0).cpu().numpy(),
                      imgTensor2.squeeze(0).permute(1,2,0).cpu().numpy(),]
            io_adapter = IOAdapter(model, images[0].shape[:2])
            inputs = io_adapter.prepare_inputs(images)
            inputs['images'] = inputs['images'].to(device)

            with torch.no_grad():

                predictions = model(inputs)
                predictions = io_adapter.unpad_and_unscale(predictions)
                flows = predictions['flows']
                flow_rgb = flow_utils.flow_to_rgb(flows)
                flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
                flow_rgb_npy = flow_rgb.detach().cpu().numpy()*255

                RGB = np.uint8(flow_rgb_npy)

                HSV = cv2.cvtColor(RGB , cv2.COLOR_RGB2HSV)
                HSVStack.append(HSV)
            
        HSVStack = np.array(HSVStack)
        HSVMeanList = [np.mean(HSVStack[max(0, i - n):min(len(HSVStack), i + n + 1)], axis=0)[:, :, 1].astype(np.uint8) for i in range(len(HSVStack))]
        
        # Calculate perfectness of the mask
        minDistance = 255
        perfectMean = HSVMeanList[0].copy()
        HM_dis_list = []
        
        for HM in HSVMeanList:
            # Normalize the mask
            HM = ((HM - np.min(HM)) / (np.max(HM) - np.min(HM)) * 255).astype(np.uint8)
            distance = 128 - np.mean(abs(128 - HM.astype(int)))
            HM_dis_list.append(distance)
            if distance < minDistance:
                minDistance = distance
                perfectMean = HM

        # Save the perfect mean
        _, thresh = cv2.threshold(np.uint8(perfectMean), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        SegMaskList.append(thresh)
        
    return SegMaskList





def getFlow(imgTensor, of_model, train_datset, ResizeFactor=2, deviceId = "cuda", n = 1):

    """
    imgList: List of image paths can be extracted from glob
    ResizeFactor: Resize factor for the image. Default is 2 times downsampled
    deviceId: Device ID for the GPU. Default is "cuda" so that it doesn't use the main GPU
    n = 5: Number of frames to consider for the optical flow. Default is 5
    """
    
    N, C, H, W = imgTensor.shape
    resized = False
    if max(imgTensor.shape) > 1000:
        resized = True
        ResizeFactor = 2
        imgTensor = F.interpolate(imgTensor, size=((H//ResizeFactor//8)*8, (W//ResizeFactor//8)*8), mode='bilinear', align_corners=False)
    
    # Resize the images by ResizeFactor to Downsample
    

    device = deviceId if torch.cuda.is_available() else "cpu"

    # model = ptlflow.get_model("pwcnet",pretrained_ckpt='things')
    model = ptlflow.get_model(of_model,pretrained_ckpt=train_datset)
    model = model.to(device)


    batch = 1
    SegMaskList = []
    imgTensor = imgTensor.to(device)

    
    for idx in tqdm(range(0, len(imgTensor))):
        HSVStack = []

        # Loop through max(i-n, 0) to min(len(dataset), i+n)
        for diff in range(-n, n + 1):
            if diff == 0 or idx + diff < 0 or idx + diff >= len(imgTensor):
                continue
            
            imgTensor1 = imgTensor[idx: idx + batch]
            imgTensor2 = imgTensor[idx + diff: idx + batch + diff]

            images = [imgTensor1.squeeze(0).permute(1,2,0).cpu().numpy(),
                      imgTensor2.squeeze(0).permute(1,2,0).cpu().numpy(),]
            io_adapter = IOAdapter(model, images[0].shape[:2])
            inputs = io_adapter.prepare_inputs(images)
            inputs['images'] = inputs['images'].to(device)

            with torch.no_grad():

                predictions = model(inputs)
                predictions = io_adapter.unpad_and_unscale(predictions)
                flows = predictions['flows']
                flow_rgb = flow_utils.flow_to_rgb(flows)
                flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
                flow_rgb_npy = flow_rgb.detach().cpu().numpy()*255

                RGB = np.uint8(flow_rgb_npy)

                HSV = cv2.cvtColor(RGB , cv2.COLOR_RGB2HSV)
                HSVStack.append(HSV)
            
        HSVStack = np.array(HSVStack)
        HSVMeanList = [np.mean(HSVStack[max(0, i - n):min(len(HSVStack), i + n + 1)], axis=0)[:, :, 1].astype(np.uint8) for i in range(len(HSVStack))]
        
        # Calculate perfectness of the mask
        minDistance = 255
        perfectMean = HSVMeanList[0].copy()
        HM_dis_list = []
        
        for HM in HSVMeanList:
            # Normalize the mask
            HM = ((HM - np.min(HM)) / (np.max(HM) - np.min(HM)) * 255).astype(np.uint8)
            distance = 128 - np.mean(abs(128 - HM.astype(int)))
            HM_dis_list.append(distance)
            if distance < minDistance:
                minDistance = distance
                perfectMean = HM

        # Save the perfect mean
        _, thresh = cv2.threshold(np.uint8(perfectMean), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resized:
            thresh = cv2.resize(thresh, (W, H))
        SegMaskList.append(thresh)
        
    return SegMaskList