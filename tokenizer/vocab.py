import numpy as np


class Vocab(object):
    def __init__(self, replace_lower_freq_word=False, replace_word="<UNK>"):
        """
        :param replace_lower_freq_word: boolean. Whether replace lower frequency and OOV word with `replace_word`.
        If False, these words removed from sequences.
        :param replace_word: str. Replacing word for OOV word.
        """
        self.word2id = {}
        self.id2word = []
        self.word2freq = {}
        self.id2freq = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word

    def add_word(self, word: str):
        """
        Update word related class attributes.
        :param word: str. Added word.
        :return: None
        """
        if word in self.word2id:
            self.word2freq[word] += 1
        else:
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
            self.word2freq[word] = 1

    def remove_low_freq_words(self, min_count=5):
        """
        Sort and then remove lower frequency words (and replace them)
        :param min_count: Threshold of word frequency.
        :return: None
        """
        self.id2word = sorted(self.word2freq, key=self.word2freq.get, reverse=True)

        for new_word_id, word in enumerate(self.id2word):
            freq = self.word2freq[word]
            if freq >= min_count:
                self.word2id[word] = new_word_id
            else:
                if self.replace_lower_freq_word:
                    self.word2id[self.replace_word] = new_word_id
                    sum_unk_freq = 0
                    for unknown_word in self.id2word[new_word_id:]:
                        sum_unk_freq += self.word2freq[unknown_word]
                        del self.word2id[unknown_word]
                    self.word2freq[self.replace_word] = sum_unk_freq
                    self.id2word = self.id2word[:new_word_id]
                    self.id2word.append(self.replace_word)
                else:
                    for unknown_word in self.id2word[new_word_id:]:
                        del self.word2id[unknown_word]
                        del self.word2freq[unknown_word]
                    self.id2word = self.id2word[:new_word_id]

                break

        self.id2freq = np.array([self.word2freq[word] for word in self.id2word])

    def __len__(self):
        return len(self.id2word)
