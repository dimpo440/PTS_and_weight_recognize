import cv2 as cv
import aimodels.yolo as yolo

# Weight class contains detection and recognition model
# Weights loading on class init
# After class init few methods are available:
# detection_model_result crops detected fields pics to list
# recognition_model_result can recognize real number on cropped image
# detect_weight is the main method to make text from any photo with weights display


YOLO_DETECT = 'model_weights/work_yolo_weight_detect.pt'
YOLO_RECOGNIZE = 'model_weights/work_yolo_weight_recognize.pt'


class Weight:
    def __init__(self, yolo_detect_weights=YOLO_DETECT, yolo_recognition_weights=YOLO_RECOGNIZE):
        self.det_model_processor = yolo.ModelLoader(weights=yolo_detect_weights).model
        self.rec_model_processor = yolo.ModelLoader(weights=yolo_recognition_weights).model

    def detection_model_result(self, img):  # the result is a list with cropped images of displays
        self.det_model_processor.iou = 0.1
        result = self.det_model_processor(img)
        return [crop['im'] for crop in result.crop(save=False)]

    def recognition_model_result(self, image):  # the result is text of the field
        # yolo inference settings
        self.rec_model_processor.conf = 0.2

        results = self.rec_model_processor(image).pandas().xyxy[0]

        # rest one dot with max confidence and sort by left to right position
        dots = results[results['class'] == 10].sort_values(by=['confidence'], ascending=[False])
        results = results.drop(dots.index[1:]).sort_values(by=['xmin'])

        # make string from symbols
        result = ''.join(map(lambda x: str(x) if x != 10 else '.', results["class"]))
        return result

    def recognize_weight(self, img_path):  # result is dictionary with fields
        fields_text = dict()
        img = cv.imread(img_path)
        fields_imgs = self.detection_model_result(img)
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = self.recognition_model_result(field_img)
        return fields_text
