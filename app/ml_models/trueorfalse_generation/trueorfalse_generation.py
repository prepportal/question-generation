import random
import numpy
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from nltk.tokenize import sent_tokenize
from typing import List

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
class TrueorFalseGenerator:
    def __init__(self,):
        model_path = "app/ml_models/true_or_false/models/BoolQ"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate(self, text: str) -> str:
        return self.model.predict(text)
    
    def beam_search_decoding (self, inp_ids,attn_mask):
        beam_output = self.model.generate(input_ids=inp_ids,
                                        attention_mask=attn_mask,
                                        max_length=256,
                                    num_beams=10,
                                    num_return_sequences=1,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True
                                    )
        Questions = [self.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                    beam_output]
        return [Question.strip().capitalize() for Question in Questions]
    
    def generate(self, context: str, count: int) -> str:
        context_splits = self._split_context_according_to_desired_count(context, count)
        question = []
        for splits in context_splits:
            answer = self.random_choice()
            text = "truefalse: %s passage: %s </s>" % (splits, answer)
            encoding = self.tokenizer.encode_plus(text, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
            output = self.beam_search_decoding(input_ids, attention_masks)
            for out in output:
                question.append({"question": out, "answer": "true" if answer == 1 else "false", "option1": "false" if answer == 1 else "true"})
        return question
    
    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        context_splits = []

        if sent_ratio < 1:
            return sents
        else:
            take_sents_count = int(sent_ratio + 1)

            start_sent_index = 0

            while start_sent_index < len(sents):
                context_split = ' '.join(sents[start_sent_index: start_sent_index + take_sents_count])
                context_splits.append(context_split)
                start_sent_index += take_sents_count - 1

        return context_splits
    
    def random_choice(self):
        a = random.choice([0,1])
        return bool(a)