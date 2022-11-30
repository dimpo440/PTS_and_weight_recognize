from PIL import Image

class STS:
    def __int__(self):
        self.yolo_weights = "best.pt"
        self.ocr_weights = ""
        self.fields_names = {0: "sign",
                             1: "vin"}

    def detect_sts(self, img_path):
        fields_text = {0: "",
                       1: ""}
        img = Image.open(img_path)
        img = rotate_sts(img)
        fields_imgs = yolo_sts_fields(img)
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = ocr_recognize(field_img)
        return fields_text

    def rotate_sts(self, img):
        return result_img

    def yolo_sts_fields(self, img):
        return fields_imgs

    def ocr_recognize(self, img):
        return text
