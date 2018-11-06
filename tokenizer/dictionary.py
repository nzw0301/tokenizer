import numpy as np
from .vocab import Vocab


class Dictionary(object):
    def __init__(
        self,
        replace_lower_freq_word=False,
        min_count=5,
        replace_word='<unk>',
        bos_word='<bos>',
        eos_word='<eos>'
    ):
        self.vocab = Vocab(replace_lower_freq_word, replace_word)
        self.num_words = 0
        self.num_vocab = 0
        self.discard_table = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.min_count = max(min_count, 1)
        self.replace_word = replace_word
        self.bos_word = bos_word
        self.eos_word = eos_word
        self.is_tokenized = False

    def _add_special_word(self, doc):
        return self.bos_word + ' ' + doc + ' ' + self.eos_word

    def _process_doc(self, doc: str):
        returned_words = []
        for word in doc.split():
            if word in self.vocab.word2id:
                returned_words.append(self.vocab.word2id.get(word))
            elif self.replace_lower_freq_word:
                returned_words.append(self.vocab.word2id.get(self.replace_word))
        return returned_words

    def fit(self, docs):
        if self.is_tokenized:
            print("Warning: this instance has already fitted.")
        is_str = isinstance(docs, str)
        if is_str:
            fname = docs
            docs = open(fname)
        elif not isinstance(docs, list):
            raise ValueError("docs is a filename of list of string")

        for doc in docs:
            doc = self._add_special_word(doc)
            for word in doc.split():
                self.vocab.add_word(word=word)
        if is_str:
            docs.close()

        self.vocab.remove_low_freq_words_from_dict(min_count=self.min_count)

        self.num_vocab = len(self.vocab)
        self.num_words = np.sum(self.vocab.id2freq)
        self.is_tokenized = True

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

    def transform(self, docs):
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")

        is_str = isinstance(docs, str)
        if is_str:
            fname = docs
            docs = open(fname)
        elif not isinstance(docs, list):
            raise ValueError("docs is a filename of list of string")

        returned_docs = []
        for doc_id, doc in enumerate(docs):
            doc = self._add_special_word(doc)
            words = self._process_doc(doc)
            if len(words) >= 1:
                returned_docs.append(np.array(words))
            else:
                print('Warning: doc {} is empty'.format(doc_id))

        if is_str:
            docs.close()
        return np.array(returned_docs)
