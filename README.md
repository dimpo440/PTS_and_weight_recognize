# PTS_and_weight_recognize
Проект по распознаванию номера ТС и его VIN с фотографии СТС. Распознавание веса детали по её фотографии на весах скоро сделаем.
На данный момент реализована только часть для обработки СТС.

Для работы проекта необходимо иметь веса для yolov5 обученной на поиск требуемых полей на фото СТС и вторые веса, которые ищут слова "Свидетельство" и "ТС" для реализации функции поворота.

Для распознавания текста в проекте встроены PaddleOCR и TRocr. Они могут работать на своих встроенных весах для распознавания печатного текста, либо можно подставить свои веса для конкретной задачи.

This project created for detection and recognition of specific fields (sign and VIN) from a photo of car registration document issued in Russian Federation. Detection and recognition of weight data from photo of weights is coming soon. 

To make this project work you need to train yolov5 for fields detection and for detection of words "svidetelstvo" and "ts" in the header of the document (for rotation use). 

For text recognition it can be used PaddleOCR model or TRocr, both of them included in the project and can be used with pretraned weights or finetuned by ourself.

<summary>Установка/Install</summary>

Clone repo and install [requirements.txt](https://github.com/dimpo440/PTS_and_weight_recognize/requirements.txt)
in a [**Python>=3.7.0**](https://www.python.org/) environment

```python
!git clone https://github.com/dimpo440/PTS_and_weight_recognize  # clone
!pip install -r PTS_and_weight_recognize/requirements.txt  # install
```

```python
YOLO_STS = 'YOUR_PATH'
YOLO_ROTATE_STS = 'YOUR_PATH'
OCR_WEIGHTS_STS = None # or 'YOUR_PATH'. Указать путь или None, чтобы использовать стандартные предобученные веса. If None pretrained weights will be used.
DETECT_MODELS = ['paddle', 'tr']
DETECT_MODEL = DETECT_MODELS[1] # Выбери PaddleOCR - 0 или TRocr - 1
TEST_IMG = 'YOUR_PATH'
DEBUG = False

test_sts = scenario.sts.STS(yolo_detect_weights=YOLO_STS, yolo_rotate_weights=YOLO_ROTATE_STS, , ocr_weights=OCR_WEIGHTS_STS)
print(test_sts.detect_sts(TEST_IMG, detect_model=DETECT_MODEL, debug=DEBUG))
```
