from PIL import Image
import cv2 as cv
import aimodels.yolo as yolo
import aimodels.paddle_ocr as po
import aimodels.tr_ocr as trtrtr
import imgprocessing.rotation as rotation


class STS:
    def __init__(self, yolo_detect_weights, yolo_rotate_weights, ocr_weights=None):
        self.yolo_detect_weights = yolo_detect_weights
        self.ocr_weights = ocr_weights
        self.yolo_rotate_weights = yolo_rotate_weights
        '''self.fields = {0: {'name': 'sign',
                           'im': None,
                           'txt': None},
                       1: {'name': 'vin',
                           'im': None,
                           'txt': None}}'''

    def detect_sts(self, img_path, detect_model='paddle', debug=False):
        fields_text = {0: "",
                       1: ""}
        img = cv.imread(img_path)  # Image.open(img_path)
        img = rotation.get_rotated(img, self.yolo_rotate_weights)
        fields_imgs = self.yolo_sts_fields(img)
        if debug:
            for img in fields_imgs:
                cv.imshow('field', img)
                k = cv.waitKey(0)
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = self.ocr_recognize(field_img, detect_model)
        return fields_text

    def yolo_sts_fields(self, img):
        model = yolo.ModelLoader(weights=self.yolo_detect_weights).model
        result = model(img)
        return [crop['im'] for crop in result.crop(save=False)]

    def ocr_recognize(self, img, detect_model):
        if detect_model=='paddle':
            model = po.ModelLoader(weights=self.ocr_weights).model
            text = model.ocr(img, det=False, rec=True, cls=False)
            return text[0][0]  # [0]
        elif detect_model=='tr':
            model = trtrtr.ModelLoader(trained_model=self.ocr_weights)
            return model.ocr(img)
        else:
            return detect_model + ' is not in the project'


