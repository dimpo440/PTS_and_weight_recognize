import scenario.sts

YOLO_STS = 'test/weights/yolo/sts_fields_best1.pt'
DETECT_MODELS = ['paddle', 'tr']
TEST_IMG = 'test/imgs/стс  Сергеев.jpg'
DEBUG = False
DETECT_MODEL = DETECT_MODELS[1]

if __name__ == '__main__':

    test_sts = scenario.sts.STS(yolo_weights=YOLO_STS)
    print(test_sts.detect_sts(TEST_IMG, detect_model=DETECT_MODEL, debug=DEBUG))
