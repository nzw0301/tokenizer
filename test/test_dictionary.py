from tokenizer import Dictionary
import numpy as np
import os
import string

text8_like_file_name = 'text8-like.txt'
doc_file_name = 'doc.txt'


def create_embeddings_files():
    with open(text8_like_file_name, 'w') as f:
        words = []
        for count, char in enumerate(string.ascii_lowercase, start=1):
            for _ in range(count):
                words.append(char)
        f.write(' '.join(words))

    with open(doc_file_name, 'w') as f:
        for count, char in enumerate(string.ascii_lowercase, start=1):
            words = []
            for _ in range(count):
                words.append(char)
            f.write(' '.join(words) + '\n')


def delete_embeddings_files():
    os.remove(text8_like_file_name)
    os.remove(doc_file_name)


def tests():
    create_embeddings_files()

    def test_fit(fname):
        dictionary = Dictionary(min_count=5, replace_lower_freq_word=False, replace_word='', bos_word='', eos_word='')
        dictionary.fit(fname)
        assert len(dictionary.vocab) == 26-4
        assert dictionary.vocab.id2word[0] == 'z'
        assert dictionary.vocab.id2word[-1] == 'e'
        assert dictionary.vocab.id2freq[0] == 26
        assert dictionary.num_words == np.sum(np.arange(5, 27))

    def test_fit_with_replace_and_add_special(fname):
        dictionary = Dictionary(min_count=5, replace_lower_freq_word=True, replace_word='<unk>', bos_word='<bos>',
                                eos_word='<eos>')
        dictionary.fit(fname)

        assert dictionary.vocab.id2word[-1] == '<unk>'
        assert dictionary.vocab.id2freq[0] == 26

        if 'text8' in fname:
            assert dictionary.vocab.id2word[0] == 'z'
            assert len(dictionary.vocab) == 26 - 4 + 1
            assert dictionary.num_vocab == 26 - 4 + 1
            assert dictionary.num_words == np.sum(np.arange(27)) + 2  # <bos> and <eos>
        else:
            assert dictionary.vocab.id2word[2] == 'z'
            assert len(dictionary.vocab) == 26 - 4 + 3
            assert dictionary.num_vocab == 26 - 4 + 3
            assert dictionary.num_words == np.sum(np.arange(1, 27) + 2)

    def test_transform(fname):
        dictionary = Dictionary(min_count=1, replace_lower_freq_word=False, bos_word='', eos_word='')
        dictionary.fit(fname)
        docs = dictionary.transform(fname)
        assert docs[0] == np.array([0])

    for fname in [text8_like_file_name, doc_file_name]:
        test_fit(fname)
        test_fit_with_replace_and_add_special(fname)

        if 'text8' in fname:
            continue

        with open(fname) as f:
            docs = f.readlines()
            test_fit(docs)
            test_fit_with_replace_and_add_special(docs)
            test_transform(docs)

    delete_embeddings_files()
