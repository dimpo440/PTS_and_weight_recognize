import scenario.sts

YOLO_STS = 'test/weights/yolo/sts_fields_best1.pt'
YOLO_ROTATE_STS = 'test/weights/yolo/sts_rotation_best-11.pt'
DETECT_MODELS = ['paddle', 'tr']
TEST_IMG = 'test/imgs/test_sts.jpg'
DEBUG = False
DETECT_MODEL = DETECT_MODELS[1]

if __name__ == '__main__':

    test_sts = scenario.sts.STS(yolo_detect_weights=YOLO_STS, yolo_rotate_weights=YOLO_ROTATE_STS)
    print(test_sts.detect_sts(TEST_IMG, detect_model=DETECT_MODEL, debug=DEBUG))
