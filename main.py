import scenario.sts
import scenario.weight
import cv2 as cv

STS_TEST_IMG = 'test/imgs/test_sts.jpg'

VES_TEST_IMG = 'test/imgs/weight/parts_used_11_44_58_22445850_11.jpg'
VES_TEST_IMG_CROPPED = 'test/imgs/weight/photo_2023-01-31_00.jpg'

if __name__ == '__main__':

    print('Что тестируем? 0 - стс, 1 - весы')
    test_choice = int(input())
    fields_text = {}
    if test_choice:
        test_ves = scenario.weight.Weight()
        print(test_ves.recognize_weight(VES_TEST_IMG))
    else:
        test_sts = scenario.sts.STS()
        img = cv.imread(STS_TEST_IMG)
        #cv.imshow('input', img)
        img = test_sts.rotation_model_result(img)
        #cv.imshow('rotated', img)
        fields_imgs = test_sts.detection_model_result(img)
        for field_img, i in fields_imgs:
            cv.imshow('field_'+str(i), field_img)
            fields_text[i] = test_sts.recognition_model_result(field_img, bool(i))
        print(fields_text)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print(test_sts.recognize_sts(STS_TEST_IMG))
