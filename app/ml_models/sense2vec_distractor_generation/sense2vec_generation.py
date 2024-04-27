import string
from sense2vec import Sense2Vec
from collections import OrderedDict
from typing import List

class Sense2VecDistractorGeneration():
    def __init__(self):
        self.s2v = Sense2Vec().from_disk('app/ml_models/sense2vec_distractor_generation/data/s2v_old')
    def edits(word):
        "All edits that are one edit away from `word`."
        letters = f'abcdefghijklmnopqrstuvwxyz {string.punctuation}'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def generate(self, word: str, desired_count: int) -> List[str]:
        output = []
        word_preprocessed = word.translate(
            word.maketrans("", "", string.punctuation))
        word_preprocessed = word_preprocessed.lower()

        word_edits = self.edits(word_preprocessed)

        word = word.replace(" ", "_")

        sense = self.s2v.get_best_sense(word)
        most_similar = self.s2v.most_similar(sense, n=15)

        compare_list = [word_preprocessed]
        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ")
            append_word = append_word.strip()
            append_word_processed = append_word.lower()
            append_word_processed = append_word_processed.translate(
                append_word_processed.maketrans("", "", string.punctuation))
            if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
                output.append(append_word.title())
                compare_list.append(append_word_processed)

        return list(OrderedDict.fromkeys(output))

