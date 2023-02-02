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

vin_symbols = "0123456789ABCDEFGHJKLMNOPRSTUVWXYZ"
plate_symbols = "0123456789ABEKMHOPCTYX"
plate_num_pos = [1, 2, 3, -2, -1]
plate_letter_pos = [0, 4, 5]
vin_num_pos = [-4, -3, -2, -1]
replace_num_char = {'8': 'B', '5': 'S', '2': 'Z', '0': 'O'}
IoU_thres = 0.4
alpha = 0.6
plate_min_len = 7
vin_min_len = 17


def formatting(field_symbols_classes, is_plate):
    allowed_symbols = ''
    text = ''
    if len(field_symbols_classes) >= plate_min_len:
        if is_plate:
            allowed_symbols = plate_symbols
            for pos in plate_letter_pos:
                for t in range(len(replace_num_char)):
                    if int(field_symbols_classes[pos]) > len(allowed_symbols) - 1:
                        break
                    if allowed_symbols[int(field_symbols_classes[pos])] == \
                            tuple(replace_num_char.items())[t][0]:
                        field_symbols_classes[pos] = str(allowed_symbols.find(tuple(replace_num_char.items())[t][1]))

            for pos in plate_num_pos:
                for t in range(len(replace_num_char)):
                    if int(field_symbols_classes[pos]) > len(allowed_symbols) - 1:
                        break
                    if allowed_symbols[int(field_symbols_classes[pos])] == \
                            tuple(replace_num_char.items())[t][1]:
                        field_symbols_classes[pos] = str(allowed_symbols.find(tuple(replace_num_char.items())[t][0]))
        else:
            allowed_symbols = vin_symbols
            for pos in vin_num_pos:
                for t in range(len(replace_num_char)):
                    if int(field_symbols_classes[pos]) > len(allowed_symbols) - 1:
                        break
                    if allowed_symbols[int(field_symbols_classes[pos])] == \
                            tuple(replace_num_char.items())[t][1]:
                        field_symbols_classes[pos] = str(allowed_symbols.find(tuple(replace_num_char.items())[t][0]))
    for element in field_symbols_classes:
        if int(element) > len(allowed_symbols) - 1:
            break
        text += allowed_symbols[int(element)]
    return text

vin_symbols = "0123456789ABCDEFGHJKLMNOPRSTUVWXYZ"
plate_symbols = "0123456789ABEKMHOPCTYX"
replace_num_char = {'8': 'B', '5': 'S', '2': 'Z', '0': 'O'}
plate_letter_pos = [0, 4, 5]

def formatting_2(elems_new, is_plate):
    all_text = plate_symbols if is_plate else vin_symbols
    text = ""
    vin_symbols = "0123456789ABCDEFGHJKLMNOPRSTUVWXYZ"
    plate_symbols = "0123456789ABEKMHOPCTYX"
    replace_num_char = {'8': 'B', '5': 'S', '2': 'Z', '0': 'O'}
    plate_letter_pos = [0, 4, 5]
    vin_num_pos = [len(elems_new)-1+p for p in [-4, -3, -2, -1]]
    for pos, element in enumerate(elems_new):
        if int(element) > len(all_text) - 1:
            break
        else:
            switched = False
        if is_plate:
            if pos in plate_letter_pos:
                for k, v in replace_num_char.items():
                    if all_text[int(element)] == k:
                        text += v
                        switched = True
                        break
            else:
                for v, k in replace_num_char.items():
                    if all_text[int(element)] == k:
                        text += v
                        switched = True
                        break
        elif pos in vin_num_pos:
            for k, v in replace_num_char.items():
                if all_text[int(element)] == v:
                    text += k
                    switched = True
                    break
        if not switched:
            text += all_text[int(element)]
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
        result = formatting_2(result['class'].values.tolist(), not cl)

        return result

    def recognize_sts(self, img_path):  # result is dictionary with fields
        fields_text = dict()
        img = cv.imread(img_path)
        img = self.rotation_model_result(img)
        fields_imgs = self.detection_model_result(img)
        for field_img, i in fields_imgs:
            fields_text[i] = self.recognition_model_result(field_img, bool(i))
        return fields_text
