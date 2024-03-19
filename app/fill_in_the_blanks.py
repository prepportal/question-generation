import string
import pke
from nltk.corpus import stopwords
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor


class Fill_In_The_Blanks:

    def generate_fill_in_the_blanks(self, text, num_questions):
        sentences = self.tokenize_sentences(text)
        keywords = self.get_noun_adj_verb(text)
        sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)
        fill_in_the_blanks = self.get_fill_in_the_blanks(sentence_mapping, num_questions)
        return fill_in_the_blanks

    def tokenize_sentences(self, text):
        sentences = sent_tokenize(text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_noun_adj_verb(self, text):
        out=[]
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text,language='en')
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'VERB', 'ADJ', 'NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            # extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_selection(pos=pos)
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=30)


            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()

        return out


    def get_sentences_for_keyword(self, keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
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
        return keyword_sentences


    def get_fill_in_the_blanks(self, sentence_mapping, num_questions):
        out={"title":"Fill in the blanks for these sentences with matching words at the top"}
        questions = []
        keys=[]
        for key in sentence_mapping:
            if len(sentence_mapping[key])>0:
                sent = sentence_mapping[key][0]
                # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
                insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
                line = insensitive_sent.sub(' _________ ', sent)
                # if (sentence_mapping[key][0] not in processed) and no_of_replacements < 2:
                if no_of_replacements < 2:
                    questions.append({"question": line, "answer":key})
        out["questions"] = questions[:num_questions]
        return out
    
fill = Fill_In_The_Blanks()
print(fill.generate_fill_in_the_blanks("The quick brown fox jumps over the lazy dog", 1))