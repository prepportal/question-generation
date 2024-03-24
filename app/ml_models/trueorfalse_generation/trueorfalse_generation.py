import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


class TrueorFalseGenerator:
    def __init__(self,):
        model_path = "app/ml_models/true_or_false/models/BoolQ"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, text: str) -> str:
        return self.model.predict(text)
    
    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def beam_search_decoding (self, inp_ids,attn_mask, count):
        beam_output = self.model.generate(input_ids=inp_ids,
                                        attention_mask=attn_mask,
                                        max_length=256,
                                    num_beams=5,
                                    num_return_sequences=count,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True
                                    )
        Questions = [self.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                    beam_output]
        return [Question.strip().capitalize() for Question in Questions]
    
    def generate(self, context: str, count: int, truefalse = "yes") -> str:
        text = "truefalse: %s passage: %s </s>" % (truefalse, context)
        encoding = self.tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        output = self.beam_search_decoding(input_ids, attention_masks, count)
        question = []
        for out in output:
            question.append({"question": out, "answer": True if truefalse == "yes" else False})
        return question