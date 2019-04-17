import numpy as np


class DiscardSampler(object):
    """
    Sub-sampling tokens for a large corpus used in word2vec/fastText.
    """
    def __init__(self, sample_t=0., rnd=np.random.RandomState(7)):
        """
        :param sample_t: sub-sampling parameter
        :param rnd: Random seed.
        """
        self.sample_t = sample_t
        self.rnd = rnd
        self.discard_table = None

    def build_discard_table(self, word_id2freq: np.ndarray):
        """
        :param word_id2freq: word id -> word frequency created by `Dictionary` instance.
        :return: None
        """
        # https://github.com/facebookresearch/fastText/blob/157e80e5775e57f9c2e0e20ac8ab6721d4771169/src/dictionary.cc#L294
        tf = self.sample_t / (word_id2freq / np.sum(word_id2freq))
        self.discard_table = np.sqrt(tf) + tf

    def discard(self, word_id: int):
        """
        :param word_id: int. Word id.
        :return: Boolean.
        True: Remove the word from its sequence.
        False: Use the word.
        """
        return self.rnd.rand() > self.discard_table[word_id]
