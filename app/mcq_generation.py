import string
import traceback
from typing import List
from nltk.tokenize import sent_tokenize
import spacy
import toolz
import pke

from app.modules.duplicate_removal import remove_distractors_duplicate_with_correct_answer, remove_duplicates
from app.modules.text_cleaning import clean_text
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.ml_models.sense2vec_distractor_generation.sense2vec_generation import Sense2VecDistractorGeneration
from app.models.question import Question

import time
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from flashtext import KeywordProcessor


class MCQGenerator():
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        # Currently not used
        # self.answer_generator = AnswerGenerator()
        # print('Loaded AnswerGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.question_generator = QuestionGenerator()
        print('Loaded QuestionGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        print('Loaded Sense2VecDistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''
        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.nlp = spacy.load('en_core_web_sm')
    # Main function
    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        cleaned_text =  clean_text(context)

        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)
        
        for question in questions:
            print('-------------------')
            print(question.answerText)
            print(question.questionText)
            print(question.distractors)

        return questions

    def _generate_answers(self, context: str, desired_count: int) -> List[Question]:
        # answers = self.answer_generator.generate(context, desired_count)
        answers = self._generate_multiple_answers_according_to_desired_count(context, desired_count)

        print(answers)
        unique_answers = remove_duplicates(answers)

        questions = []
        for answer in unique_answers:
            questions.append(Question(answer))

        return questions

    def _generate_questions(self, context: str, questions: List[Question]) -> List[Question]:        
        for question in questions:
            question.questionText = self.question_generator.generate(question.answerText, context)

        return questions

    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        # context_splits = self._split_context_according_to_desired_count(context, desired_count)

        # questions = []

        # for split in context_splits:
        #     keywords = self._get_noun_adj_verb(split)
        #     for key in keywords:
        #         if len(questions) >= desired_count:
        #             break
        #         if self.sense2vec_distractor_generator.MCQs_available(key):
        #             answer, question = self.question_generator.generate_qna(split, key)
        #             questions.append(Question(answer.capitalize(), question))

        # questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        # return questions
        sentences = self.tokenize_sentences(context)
        joiner = " "
        modified_text = joiner.join(sentences)
        keywords = self.get_keywords(self.nlp,modified_text,desired_count,self.fdist,self.normalized_levenshtein,len(sentences) )
        keyword_sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet
        questions = []

        if len(keyword_sentence_mapping.keys()) == 0:
            return []
        else:
            answers = keyword_sentence_mapping.keys()
            for answer in answers:
                txt = keyword_sentence_mapping[answer]
                ans, question = self.question_generator.generate_qna(txt, answer)
                questions.append(Question(ans, question))
        questions = list(toolz.unique(questions, key=lambda x: x.answerText))
        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        for question in questions:
            distractors = self.sense2vec_distractor_generator.generate(question.answerText)


            # distractors = remove_duplicates(distractors)
            # distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)
            #TODO - filter distractors having a similar bleu score with another distractor

            question.distractors = distractors

        return questions

    # Helper functions 
    def _generate_answer_for_each_sentence(self, context: str) -> List[str]:
        sents = sent_tokenize(context)

        answers = []
        for sent in sents:
            answers.append(self.answer_generator.generate(sent, 1)[0])

        return answers

    #TODO: refactor to create better splits closer to the desired amount
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

    def _get_noun_adj_verb(self, text):
        out = []
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text, language='en')
        pos = {'PROPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        try:
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
        except:
            return out

        keyphrases = extractor.get_n_best(n=10)

        for key in keyphrases:
            out.append(key[0])

        return out
    
    def get_keywords(self, nlp,text,max_keywords,fdist,normalized_levenshtein,no_of_sentences):
        doc = nlp(text)
        max_keywords = int(max_keywords)

        keywords = self._get_noun_adj_verb(text)
        keywords = sorted(keywords, key=lambda x: fdist[x])
        keywords = self.filter_phrases(keywords, max_keywords,normalized_levenshtein )

        phrase_keys = self.get_phrases(doc)
        filtered_phrases = self.filter_phrases(phrase_keys, max_keywords,normalized_levenshtein )

        total_phrases = keywords + filtered_phrases

        total_phrases_filtered = self.filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )


        answers = []
        for answer in total_phrases_filtered:
            if answer not in answers and self.sense2vec_distractor_generator.MCQs_available(answer):
                answers.append(answer)

        answers = answers[:max_keywords]
        return answers
    
    def filter_phrases(self,phrase_keys,max,normalized_levenshtein ):
        filtered_phrases =[]
        if len(phrase_keys)>0:
            filtered_phrases.append(phrase_keys[0])
            for ph in phrase_keys[1:]:
                if self.is_far(filtered_phrases,ph,0.7,normalized_levenshtein ):
                    filtered_phrases.append(ph)
                if len(filtered_phrases)>=max:
                    break
        return filtered_phrases
    
    def get_phrases(self,doc):
        phrases={}
        for np in doc.noun_chunks:
            phrase =np.text
            len_phrase = len(phrase.split())
            if len_phrase > 1:
                if phrase not in phrases:
                    phrases[phrase]=1
                else:
                    phrases[phrase]=phrases[phrase]+1

        phrase_keys=list(phrases.keys())
        phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)
        phrase_keys=phrase_keys[:50]
        return phrase_keys
    
    def is_far(self,words_list,currentword,thresh,normalized_levenshtein):
        threshold = thresh
        score_list =[]
        for word in words_list:
            score_list.append(normalized_levenshtein.distance(word.lower(),currentword.lower()))
        if min(score_list)>=threshold:
            return True
        else:
            return False

    def tokenize_sentences(self, text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        # Remove any short sentences less than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences
    
    def get_sentences_for_keyword(self, keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            word = word.strip()
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values

        delete_keys = []
        for k in keyword_sentences.keys():
            if len(keyword_sentences[k]) == 0:
                delete_keys.append(k)
        for del_key in delete_keys:
            del keyword_sentences[del_key]

        return keyword_sentences