#https://www.youtube.com/watch?v=Pb3opEFP94U
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import cv2
import numpy as np


class Detector:
    def choose_model_type(self):
        if(self.model_type == "OS"):
            self.path_config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

        elif(self.model_type == "LVIS"):
            self.path_config = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"

        elif(self.model_type == "IS"):#instance segmentation
            self.path_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        elif(self.model_type == "KP"):
            self.path_config = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        
        elif(self.model_type == "PS"):#panoptic segmentation
            self.path_config = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        print(self.path_config)

    def __init__(self, model_type = "PS" ):
        self.cfg = get_cfg()       
        self.model_type = model_type

        self.choose_model_type()#set path_config variable

        self.cfg.merge_from_file(model_zoo.get_config_file(self.path_config))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.path_config)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" #cpu or cuda
        self.predicator = DefaultPredictor(self.cfg)
        print("end init")
    

    #process images functions, panoptic or not
    def process_not_PS(self, image):
        predictions = self.predicator(image)
        viz = Visualizer(image[:, :, ::-1], 
                        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                        instance_mode = ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        return(output)

    def process_PS(self, image):
            predictions,segmentInfo = self.predicator(image)["panoptic_seg"]
            viz = Visualizer(image[:, :, ::-1], 
                            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            print(segmentInfo)
            self.isStairs(segmentInfo)
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentInfo)#draw_panoptic_seg_predictions
            return(output)

    #static picture or videos
    def onImage(self, imagePath):
        image = cv2.imread(imagePath)

        if(self.model_type != "PS"):
            output = self.process_not_PS(image)

        else:
            output = self.process_PS(image)
        img_out = output.get_image()[:,:,::-1]
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        #print(img_out[400,400]) 
        self.detect_purple(output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    def thresh(self, img):
        ret, thresh1 = cv2.threshold(img,100,200,cv2.THRESH_BINARY)
        cv2.imshow("Result",thresh1)
        cv2.waitKey(0)
    
    def check_purple(self, value):#check if the value is inside the threshold of purple
        low_pruple=np.array([60, 50, 60])
        high_purple= np.array([120, 110, 1120])
        if(low_pruple[0] < value[0] < high_purple[0]):
            if(low_pruple[1] < value[1] < high_purple[1]):
                if(low_pruple[2] < value[2] < high_purple[2]):
                    return(True)
        return(False)

    def detect_purple(self, img):
        totest= img[400, 400]
        print(self.check_purple(totest))
        

    def detectpurple0(self, img):# 116 101 115 // 81 66 82
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          #transform the image in HSV
        low_pruple=np.array([60, 50, 60])
        high_purple= np.array([120, 110, 1120])
        mask = cv2.inRange(hsv, low_pruple, high_purple)
        result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("DETECTPURPLE",result)
        cv2.imshow("mask",mask)
        cv2.waitKey(0)


      
    def onCam(self, videoPath = 0):
        vid = cv2.VideoCapture(videoPath)
        if(vid.isOpened()== False):
            print("Error")
            return(None)

        else:
            ret, image = vid.read()
            while(ret):#While the video can be read#can repleace the next lines by onImage ?
                if(self.model_type != "PS"):
                    output = self.process_not_PS(image)
                else:
                    output = self.process_PS(image)

                cv2.imshow('frame', output.get_image()[:,:,::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, image = vid.read()
    
    #Print True or False if there are stairs // only work with PS setup
    def isStairs(self, segmentInfo):
        #stairs is 27
        tab = []
        for cell in segmentInfo:
            tab.append(cell['category_id'])
        if(27 in tab):
            print(True)
        else:
            print(False)

    def test(self):
        print("oui tout va bien")


#cd Documents/MSc_project/robot_vision/Detectortry
if __name__ == '__main__':
    yo = Detector("PS")
    im_file = "images/stairs1.jpg"
    yo.onImage(im_file)

    #yo.onImage("images/test1.jpg")
    # yo.onImage("images/stairs2.jpg")
    # yo.onImage("images/stairs3.jpg")
    #yo.onCam()
    print()


