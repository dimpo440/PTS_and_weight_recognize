import scenario.sts
import scenario.vesy

YOLO_STS = 'test/weights/yolo/sts_fields_best1.pt'
YOLO_ROTATE_STS = 'test/weights/yolo/sts_rotation_best-11.pt'
DETECT_MODELS = ['paddle', 'tr']
STS_TEST_IMG = 'test/imgs/test_sts.jpg'
DETECT_MODEL = DETECT_MODELS[0]
YOLO_VES_DETECTION = 'test/weights/yolo/ves_detection_best-5.pt'
YOLO_VES_RECOGNITION = 'test/weights/yolo/ves_recognition_best _11_ilya.pt'
VES_TEST_IMG = 'test/imgs/vesy/parts_used_11_44_58_22445850_11.jpg'

if __name__ == '__main__':

    print('Что тестируем? 0 - стс, 1 - весы')
    test_choice = int(input())
    if test_choice:
        test_ves = scenario.vesy.Ves(yolo_detect_weights=YOLO_VES_DETECTION,
                                     yolo_recognition_weights=YOLO_VES_RECOGNITION)
        print(test_ves.detect_ves(VES_TEST_IMG))
    else:
        test_sts = scenario.sts.STS(yolo_detect_weights=YOLO_STS,
                                    yolo_rotate_weights=YOLO_ROTATE_STS,
                                    recognition_model=DETECT_MODEL)
        print(test_sts.detect_sts(STS_TEST_IMG))
