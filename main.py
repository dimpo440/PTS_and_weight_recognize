import scenario.sts
import scenario.weight

STS_TEST_IMG = 'test/imgs/test_sts.jpg'

RECOGNITION_MODELS = ['paddle', 'tr']
RECOGNITION_MODEL = RECOGNITION_MODELS[0]
VES_TEST_IMG = 'test/imgs/weight/parts_used_11_44_58_22445850_11.jpg'

if __name__ == '__main__':

    print('Что тестируем? 0 - стс, 1 - весы')
    test_choice = int(input())
    if test_choice:
        test_ves = scenario.weight.Weight()
        print(test_ves.detect_weight(VES_TEST_IMG))
    else:
        test_sts = scenario.sts.STS(recognition_model=RECOGNITION_MODEL)
        print(test_sts.recognize_sts(STS_TEST_IMG))
