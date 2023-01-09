import cv2 as cv
import aimodels.yolo as yolo


class Weight:
    def __init__(self, yolo_detect_weights, yolo_recognition_weights):
        self.det_model_processor = yolo.ModelLoader(weights=yolo_detect_weights).model
        self.det_model = self.detection_model_results
        self.rec_model_processor = yolo.ModelLoader(weights=yolo_recognition_weights).model
        self.rec_model = self.recognition_model_result

    def detection_model_results(self, img):  # the result is a list with cropped images of fields
        result = self.det_model_processor(img)
        return [crop['im'] for crop in result.crop(save=False)]

    def recognition_model_result(self, image):  # the result is text of the field
        self.rec_model_processor.conf = 0.1
        self.rec_model_processor.conf_thres = 0.7
        results = self.rec_model_processor(image).pandas().xyxy[0].sort_values(by=["xmin"])
        result = ''.join(map(lambda x: str(x) if x != 10 else '.', results["class"]))
        return result

    def detect_weight(self, img_path):  # result is dictionary with fields
        fields_text = dict()
        img = cv.imread(img_path)
        fields_imgs = self.det_model(img)
        for i, field_img in enumerate(fields_imgs):
            fields_text[i] = self.rec_model(field_img)
        return fields_text
