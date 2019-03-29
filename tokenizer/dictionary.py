import numpy as np
import logging

from .vocab import Vocab


class Dictionary(object):
    LINE_BREAK_WORD = "</s>"

    def __init__(
        self,
        replace_lower_freq_word=False,
        min_count=5,
        replace_word='<UNK>',
        max_sentence_length=1000
    ):
        """
        :param replace_lower_freq_word: boolean. Whether replace lower frequency and OOV word with `replace_word`.
        If False, these words removed from sequences.
        :param min_count: threshold of word frequency.
        :param replace_word: str. Replacing word for OOV word.
        :param max_sentence_length: maximum word sequence length for generator version.
        """
        self.vocab = Vocab(replace_lower_freq_word, replace_word)
        self.num_words = 0
        self.num_vocab = 0
        self.discard_table = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.min_count = max(min_count, 1)
        self.replace_word = replace_word
        self.is_tokenized = False
        self.max_sentence_length = int(max_sentence_length)

        # check arguments
        assert max_sentence_length > 0, "`max_sentence_length` must be positive."

        # add special words to vocab
        if self.replace_lower_freq_word and self.replace_word:
            self.vocab.id2word.append(self.replace_word)
            self.vocab.word2id[self.replace_word] = len(self.vocab.word2id)

        # init logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        self.logger = logger

    def _process_doc(self, doc: str):
        returned_words = []
        for word in doc.split():
            if word in self.vocab.word2id:
                returned_words.append(self.vocab.word2id.get(word))
            elif self.replace_lower_freq_word:
                returned_words.append(self.vocab.word2id.get(self.replace_word))
        return returned_words

    def _read_word_generator(self, file):
        """
        :param file: file instance.
        :return: one word (str)
        """
        word = ""
        while True:
            character = file.read(1)
            if character == " " or character == "\n" or character == "\t":
                if len(word) != 0:
                    yield word
                if character == "\n":
                    yield self.LINE_BREAK_WORD
                word = ""
            # EoF
            elif character == "":
                if len(word) != 0:
                    yield word
                break
            else:
                word += character

    def _build_vocab_from_file(self, fname: str):
        """
        :param fname: str. target file name.
        :return: None
        """
        with open(fname) as f:
            for word in self._read_word_generator(f):
                if word != self.LINE_BREAK_WORD:
                    self.vocab.add_word(word=word)

    def _update_instance_values(self):
        """
        Update class attributes related to fitting corpus.
        :return: None
        """
        self.vocab.remove_low_freq_words(min_count=self.min_count)
        self.num_vocab = len(self.vocab)
        self.num_words = np.sum(self.vocab.id2freq)
        self.is_tokenized = True

    def fit_from_fname(self, fname: str, in_memory=False):
        """
        :param fname: str. Target file name.
        :param in_memory: If True, read a file on memory, then fit on it.
        else read/fit each word from file to reduce the memory usage.
        :return: None
        """
        if self.is_tokenized:
            self.logger.warning("Warning: this instance has already fitted.")

        if in_memory:
            with open(fname) as f:
                docs = f.readlines()
                self.fit_from_list(docs=docs)
        else:
            self._build_vocab_from_file(fname=fname)
            self._update_instance_values()

    def fit_from_list(self, docs):
        """
        Fit on list of str
        :param docs: List of str
        :return: None
        """
        if self.is_tokenized:
            self.logger.warning("Warning: this instance has already fitted.")

        for doc in docs:
            for word in doc.split():
                self.vocab.add_word(word=word)

        self._update_instance_values()

    def fit(self, docs, in_memory=False):
        """
        :param docs: list of str or str (file path and name)
        :param in_memory: Boolean flag. If a file is large, you should pass `True`.
            This argument is valid when type of `docs` is str.
            True: read each character like `word2vec.c`,
            False: fit a docs in-memory.
        :return: None
        """
        if self.is_tokenized:
            self.logger.warning("Warning: this instance has already fitted.")

        if isinstance(docs, list):
            self.fit_from_list(docs)
        elif isinstance(docs, str):
            self.fit_from_fname(fname=docs, in_memory=in_memory)

    def fit_transform(self, docs):
        """
        In-memory `fit` and `transform`
        :param docs: list of str or str (file path and name).
        :return: list of list of int.
        """
        self.fit(docs, in_memory=True)
        return self.transform(docs)

    # in-memory version
    def generator_transform_from_fname(self, fname: str):
        """
        Generator to read words from a large text file.
        :param fname: file names
        :return: generator word id's sequences
        """
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")
        word_ids = []
        with open(fname) as f:
            for word in self._read_word_generator(f):
                if word == self.LINE_BREAK_WORD:
                    if len(word_ids) != 0:
                        yield word_ids
                        word_ids = []

                elif word in self.vocab.word2id:
                    word_id = self.vocab.word2id[word]
                    word_ids.append(word_id)

                elif self.replace_lower_freq_word:
                    word_id = self.vocab.word2id[self.replace_word]
                    word_ids.append(word_id)

                elif word != self.LINE_BREAK_WORD:  # skip unknown word without replacement by a special word.
                    continue

                if len(word_ids) == self.max_sentence_length:
                    yield word_ids
                    word_ids = []

            if len(word_ids) != 0:
                yield word_ids

    def transform_from_list(self, docs):
        """
        :param docs: list of str
        :return: np.ndarray of np.ndarray of int. word id's sequneces
        """
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")
        returned_docs = []
        for doc_id, doc in enumerate(docs):
            words = self._process_doc(doc)
            if len(words) >= 1:
                returned_docs.append(np.array(words))
            else:
                logging.warning("Warning: {}th doc is empty".format(doc_id))
        return np.array(returned_docs)

    def transform(self, docs):
        """
        :param docs: list of str or str (file path and name)
        :return: np.ndarray of np.ndarray of int. word id's sequneces
        """
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")

        if not isinstance(docs, str) and not isinstance(docs, list):
            raise ValueError("`docs` is a filename or list of string.")

        if isinstance(docs, str):
            with open(docs) as f:
                docs = f.readlines()

        return self.transform_from_list(docs)
