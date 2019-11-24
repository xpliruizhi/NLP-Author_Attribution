from nltk.corpus import gazetteers, names
from nltk import word_tokenize
from Corpus import WikiCorpus

##......add......##
import nltk
nltk.download('gazetteers')
nltk.download('names')
#####------------#####

class DocSelection:
    def __init__(self, corpus):
        self.places = set(gazetteers.words())
        self.people = set(names.words())
        self.stop_words = self.load_stop_words()
        self.corpus = corpus

    def load_stop_words(self):
        stop_words = set()
        with open("stoplist") as f:
            for word in f:
                word=word.rstrip("\n")
                stop_words.add(word)
        return stop_words


    def select_docs(self, claim):
        select_docs = []
        claim_tokens = word_tokenize(claim)

        claim_tokens = [token.lower() for token in claim_tokens if token not in self.stop_words]

        for title in self.corpus:
            complete_match = True
            title_tokens, other_txt = WikiCorpus.normalize_title(title, rflag=True)

            for title_token in title_tokens:
                if title_token.lower() not in claim_tokens:
                    complete_match = False
                    break

            if complete_match:
                select_docs.append(title)

        return select_docs
