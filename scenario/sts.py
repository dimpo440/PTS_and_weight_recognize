from PIL import Image
import aimodels.yolo as yolo


class STS:
    def __init__(self, yolo_weights, ocr_weights=None):
        self.yolo_weights = yolo_weights
        self.ocr_weights = ocr_weights
        self.fields = {0: {'name': 'sign',
                           'im': None,
                           'txt': None},
                       1: {'name': 'vin',
                           'im': None,
                           'txt': None}}

    def detect_sts(self, img_path, debug=False):
        fields_text = {0: "",
                       1: ""}
        img = Image.open(img_path)
        img = self.rotate_sts(img)
        fields_imgs = self.yolo_sts_fields(img)
        if debug:
            for i in range(len(fields_imgs)):
                print(f'Field picture {i+1} recieved')
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = self.ocr_recognize(field_img)
        return fields_text

    def rotate_sts(self, img):
        return img

    def yolo_sts_fields(self, img):
        model = yolo.ModelLoader(weights=self.yolo_weights).model
        result = model(img)
        return [crop['im'] for crop in result.crop(save=False)]

    def ocr_recognize(self, img):
        text = 'recognizing not available'
        return text
