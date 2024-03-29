[Русский](https://github.com/dimpo440/PTS_and_weight_recognize/blob/main/README_RU.md)

# Data recognition on documents photo and mesurings data on photo
There are two scenarios in project:
- detection and recognition of specific fields (sign and VIN) from a photo
- detection and recognition of weight data from photo of weights

To make this project work you need to train yolov5 for fields detection and for detection of words "svidetelstvo" and "ts" in the header of the document (for rotation use). 

For text recognition it can be used PaddleOCR model or TRocr, both of them included in the project and can be used with pretraned weights or finetuned by ourself.

<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/dimpo440/PTS_and_weight_recognize/blob/main/requirements.txt)
in a [**Python>=3.7.0**](https://www.python.org/) environment

```python
!git clone https://github.com/dimpo440/PTS_and_weight_recognize  # clone
!pip install git+https://github.com/dimpo440/PTS_and_weight_recognize  # install
```
Example

```python
import scenario.sts

YOLO_STS = 'YOUR_PATH'
YOLO_ROTATE_STS = 'YOUR_PATH'
OCR_WEIGHTS_STS = None  # or 'YOUR_PATH'. If None pretrained weights will be used.
DETECT_MODELS = ['paddle', 'tr']
DETECT_MODEL = DETECT_MODELS[1]  # Choose PaddleOCR - 0 or TRocr - 1
TEST_IMG = 'YOUR_PATH'
DEBUG = False

test_sts = scenario.sts.STS(yolo_detect_weights=YOLO_STS,
                            yolo_rotate_weights=YOLO_ROTATE_STS,, ocr_weights = OCR_WEIGHTS_STS)
print(test_sts.recognize_sts(TEST_IMG, detect_model=DETECT_MODEL, debug=DEBUG))
```
