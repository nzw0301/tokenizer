import numpy as np


class Vocab(object):
    def __init__(self, replace_lower_freq_word=False, replace_word='<unk>'):
        self.word2id = {}
        self.id2word = []
        self.word2freq = {}
        self.id2freq = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
            self.word2freq[word] = 1
        else:
            self.word2freq[word] += 1

    def remove_low_freq_words_from_dict(self, min_count=5):
        self.id2word = sorted(self.word2freq, key=self.word2freq.get, reverse=True)

        for new_word_id, word in enumerate(self.id2word):
            freq = self.word2freq[word]
            if freq >= min_count:
                self.word2id[word] = new_word_id
            else:
                if self.replace_lower_freq_word:
                    self.word2id[self.replace_word] = new_word_id
                    sum_unk_freq = 0
                    for word in self.id2word[new_word_id:]:
                        sum_unk_freq += self.word2freq[word]
                        del self.word2id[word]
                    self.word2freq[self.replace_word] = sum_unk_freq
                    self.id2word = self.id2word[:new_word_id]
                    self.id2word.append(self.replace_word)
                else:
                    for word in self.id2word[new_word_id:]:
                        del self.word2id[word]
                    self.id2word = self.id2word[:new_word_id]

                break
        self.id2freq = np.array([self.word2freq[word] for word in self.id2word])

    def __len__(self):
        return len(self.id2word)
