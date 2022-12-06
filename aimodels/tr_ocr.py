from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class ModelLoader:
    def __init__(self, trained_model=None):
        if trained_model:
            self.processor = TrOCRProcessor.from_pretrained(trained_model)
            self.model = VisionEncoderDecoderModel.from_pretrained(trained_model)
        else:
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

    def ocr(self, image):
        # image = Image.open(img).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text