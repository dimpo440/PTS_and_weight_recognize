# Распознавание данных из фото документов и данных прибора на фото 
В проекте два сценария:
- по распознаванию номера ТС и его VIN с фотографии СТС
- распознавание веса детали по её фотографии

Для работы проекта необходимо иметь веса для yolov5 обученной на поиск требуемых полей и символов.

<summary>Установка</summary>

Clone repo and install [requirements.txt](https://github.com/dimpo440/PTS_and_weight_recognize/requirements.txt)
in a [**Python>=3.7.0**](https://www.python.org/) environment

```python
!git clone -b ru_yolo_only https://github.com/dimpo440/PTS_and_weight_recognize  # clone
!pip install -r requirements.txt  # install
```
После этого в папку model_weights необходимо распаковать архив с весами моделей
work_yolo_weight_detect.pt

work_yolo_weight_recognize.pt

work_yolo_sts_detect.pt

work_yolo_sts_rotate.pt

work_yolo_sts_vin_recognize.pt

work_yolo_sts_sign_recognize.pt

Архив доступен [по ссылке](https://drive.google.com/file/d/1uo_ubaCNT-f5N-jT7kzY5lV-grW5Kkmc/view?usp=share_link)

Пример

```python
import scenario.sts
import scenario.weight

TEST_IMG = 'YOUR_PATH.jpg'

test_sts = scenario.sts.STS()
print(test_sts.recognize_sts(TEST_IMG))

test_weight = scenario.weight.Weight()
print(test_weight.recognize_weight(TEST_IMG))
```
