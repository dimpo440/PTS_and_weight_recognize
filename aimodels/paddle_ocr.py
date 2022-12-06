from paddleocr import PaddleOCR


class ModelLoader:
    def __init__(self, weights=None, dictionary=None):
        self.weights = weights
        self.dictionary = dictionary

        if weights:
            self.model = self.load_finetuned()
        else:
            self.model = self.load_pretrained()

    def load_finetuned(self):
        model = PaddleOCR(rec_model_dir=self.weights,
                          rec_char_dict_path=self.dictionary,
                          lang='en',
                          use_space_char=False)
        return model

    def load_pretrained(self):
        model = PaddleOCR(lang='en')
        return model
