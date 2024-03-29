import cv2 as cv
import aimodels.yolo as yolo
import aimodels.paddle_ocr as po
import aimodels.tr_ocr as trtrtr
import imgprocessing.rotation as rotation

# STS class contains detection and recognition model
# Weights for them are loading on init of class
# After class init several methods are available:
# rotation_model_result will place document horizontally
# detection_model_result crops detected fields to list
# recognize_sts is the main method to make text from any photo of sts document


YOLO_ROTATE = 'model_weights/work_yolo_sts_rotate.pt'
YOLO_DETECT = 'model_weights/work_yolo_sts_detect.pt'
YOLO_RECOGNIZE = 'model_weights/work_yolo_sts_recognize.pt'


class STS:
    def __init__(self,
                 yolo_detect_weights=YOLO_DETECT,
                 yolo_rotate_weights=YOLO_ROTATE,
                 recognition_model='paddle',
                 ocr_weights=None):
        self.rot_model_processor = yolo.ModelLoader(weights=yolo_rotate_weights).model
        self.rot_model = self.rotation_model_result
        self.det_model_processor = yolo.ModelLoader(weights=yolo_detect_weights).model
        self.det_model = self.detection_model_results
        if recognition_model == 'paddle':
            self.rec_model_processor = po.ModelLoader(weights=ocr_weights).model
            self.rec_model = self.paddle_model_result
        elif recognition_model == 'tr':
            self.rec_model_processor = trtrtr.ModelLoader(trained_model=ocr_weights)
            self.rec_model = self.trocr_model_result

    def rotation_model_result(self, img):  # the result is rotated photo with sts
        return rotation.get_rotated(img, self.rot_model_processor)

    def detection_model_results(self, img):  # the result is a list with cropped images of fields
        result = self.det_model_processor(img)
        return [crop['im'] for crop in result.crop(save=False)]

    def paddle_model_result(self, img):  # the result is text of the field
        text = self.rec_model_processor.ocr(img, det=False, rec=True, cls=False)
        return text[0][0]

    def trocr_model_result(self, img):  # the result is text of the field
        return self.rec_model_processor.ocr(img)

    def recognize_sts(self, img_path):  # result is dictionary with fields
        fields_text = dict()
        img = cv.imread(img_path)
        img = self.rot_model(img)
        fields_imgs = self.det_model(img)
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = self.rec_model(field_img)
        return fields_text
