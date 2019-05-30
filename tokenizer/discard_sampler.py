import numpy as np


class DiscardSampler(object):
    """
    Sub-sampling tokens for a large corpus used in word2vec/fastText.
    This method is proposed in Sec. 2.3 of Distributed Representations of Words and Phrases and their Compositionality
    by T. Mikolov et al.
    Ref: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality
    """
    def __init__(self, sample_t=0., rnd=np.random.RandomState(7)):
        """
        :param sample_t: sub-sampling parameter
        :param rnd: numpy's `RandomState` instance.
        """
        self.sample_t = sample_t
        self.rnd = rnd
        self.discard_table = None

    def build_discard_table(self, word_id2freq: np.ndarray):
        """
        :param word_id2freq: word id -> word frequency created by `Dictionary` instance.
        :return: None
        """
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
