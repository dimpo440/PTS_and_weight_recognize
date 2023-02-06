import cv2 as cv
import aimodels.yolo as yolo
import imgprocessing.rotation as rotation

# STS class contains detection and recognition model
# Weights for them are loading on class init
# After class init several methods are available:
# rotation_model_result will place document horizontally
# detection_model_result crops detected fields to list
# recognize_sts is the main method to make text from any photo of sts document


YOLO_ROTATE = 'model_weights/work_yolo_sts_rotate.pt'
YOLO_DETECT = 'model_weights/work_yolo_sts_detect.pt'
YOLO_RECOGNIZE_VIN = 'model_weights/work_yolo_sts_vin_recognize.pt'
YOLO_RECOGNIZE_SIGN = 'model_weights/work_yolo_sts_sign_recognize.pt'


def formatting(field_symbol_classes, is_plate):
    vin_symbols = "0123456789ABCDEFGHJKLMNOPRSTUVWXYZ"
    plate_symbols = "0123456789ABEKMHOPCTYX"
    allowed_symbols = plate_symbols if is_plate else vin_symbols
    text = ""
    replace_num_char = {'8': 'B', '5': 'S', '2': 'Z', '0': 'O'}
    plate_letter_pos = [0, 4, 5]
    vin_num_pos = [len(field_symbol_classes) - 1 + p for p in [-4, -3, -2, -1]]

    for pos, element in enumerate(field_symbol_classes):
        if int(element) > len(allowed_symbols) - 1:
            break
        else:
            switched = False
        if is_plate:
            pass
        elif pos in vin_num_pos:
            for k, v in replace_num_char.items():
                if allowed_symbols[int(element)] == v:
                    text += k
                    switched = True
                    break
        if not switched:
            text += allowed_symbols[int(element)]
    return text


class STS:
    def __init__(self,
                 yolo_detect_weights=YOLO_DETECT,
                 yolo_rotate_weights=YOLO_ROTATE,
                 yolo_vin_recognize_weights=YOLO_RECOGNIZE_VIN,
                 yolo_sign_recognize_weights=YOLO_RECOGNIZE_SIGN):
        self.rot_model_processor = yolo.ModelLoader(weights=yolo_rotate_weights).model
        self.det_model_processor = yolo.ModelLoader(weights=yolo_detect_weights).model
        self.vin_rec_model_processor = yolo.ModelLoader(weights=yolo_vin_recognize_weights).model
        self.sign_rec_model_processor = yolo.ModelLoader(weights=yolo_sign_recognize_weights).model

    def rotation_model_result(self, img):  # the result is rotated photo with sts
        return rotation.get_rotated(img, self.rot_model_processor)

    def detection_model_result(self, img):  # the result is a list with cropped images of fields and classes
        result = self.det_model_processor(img)
        return [[crop['im'], int(crop['cls'].item())] for crop in result.crop(save=False)]

    def recognition_model_result(self, image, cl):
        if cl:
            yolo_model = self.vin_rec_model_processor
        else:
            yolo_model = self.sign_rec_model_processor
        yolo_model.conf = 0.5
        yolo_model.iou = 0.5
        yolo_model.agnostic = True

        result = yolo_model(image).pandas().xyxy[0].sort_values('xmin')
        result = formatting(result['class'].values.tolist(), not cl)

        return result

    def recognize_sts(self, img_path):  # result is dictionary with fields
        fields_text = dict()
        img = cv.imread(img_path)
        img = self.rotation_model_result(img)
        fields_imgs = self.detection_model_result(img)
        for field_img, i in fields_imgs:
            fields_text[i] = self.recognition_model_result(field_img, bool(i))
        return fields_text
