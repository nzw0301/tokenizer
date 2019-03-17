from tokenizer import Dictionary
import numpy as np
import os
import string

TEXT8_LIKE_FILE_NAME = 'text8-like.txt'
DOC_FILE_NAME = 'doc.txt'


def create_test_corpus_files():
    with open(TEXT8_LIKE_FILE_NAME, 'w') as f:
        """
        file content: " a b b c c c ... z" 
        """
        words = [" "]
        for count, char in enumerate(string.ascii_lowercase, start=1):
            for _ in range(count):
                words.append(char)
        f.write(' '.join(words))

    with open(DOC_FILE_NAME, 'w') as f:
        """
        file content:
         
        a
        
        b b
        
        c c c
        
        ...
        
        z ...... z
        
        """
        for count, char in enumerate(string.ascii_lowercase, start=1):
            words = []
            for _ in range(count):
                words.append(char)
            f.write('\n')
            f.write(' '.join(words) + '\n')
        f.write('\n')


def delete_test_corpus_files():
    os.remove(TEXT8_LIKE_FILE_NAME)
    os.remove(DOC_FILE_NAME)


def tests():
    def test_fit_from_fname(fname):
        for in_memory in [False, True]:
            dictionary = Dictionary(min_count=5, replace_lower_freq_word=False, replace_word='')
            dictionary.fit_from_fname(fname, in_memory=in_memory)
            assert len(dictionary.vocab) == 26-4
            assert dictionary.vocab.id2word[0] == 'z'
            assert dictionary.vocab.id2word[-1] == 'e'
            assert dictionary.vocab.id2freq[0] == 26
            assert dictionary.num_words == np.sum(np.arange(5, 27))

    def test_fit_from_docs(docs):
        dictionary = Dictionary(min_count=5, replace_lower_freq_word=False, replace_word='')
        dictionary.fit(docs)
        assert len(dictionary.vocab) == 26-4
        assert dictionary.vocab.id2word[0] == 'z'
        assert dictionary.vocab.id2word[-1] == 'e'
        assert dictionary.vocab.id2freq[0] == 26
        assert dictionary.num_words == np.sum(np.arange(5, 27))

    def test_fit_min_count1(docs):
        dictionary = Dictionary(min_count=1, replace_lower_freq_word=False, replace_word='')
        dictionary.fit(docs)
        assert len(dictionary.vocab) == 26
        assert dictionary.vocab.id2word[0] == 'z'
        assert dictionary.vocab.word2id['z'] == 0
        assert dictionary.vocab.id2word[-1] == 'a'
        assert dictionary.vocab.id2freq[0] == 26
        assert dictionary.num_words == np.sum(np.arange(1, 27))

    def test_fit_with_replace_and_add_special_from_fname(fname):
        dictionary = Dictionary(min_count=5, replace_lower_freq_word=True, replace_word='<UNK>')
        dictionary.fit(fname)

        assert dictionary.vocab.id2word[-1] == '<UNK>'
        assert dictionary.vocab.id2freq[0] == 26
        assert dictionary.vocab.id2word[0] == 'z'
        assert len(dictionary.vocab) == 26 - 4 + 1
        assert dictionary.num_vocab == 26 - 4 + 1
        assert dictionary.num_words == np.sum(np.arange(27))

    def test_transform(docs):
        dictionary = Dictionary(min_count=1, replace_lower_freq_word=False)
        dictionary.fit(docs)
        docs = dictionary.transform(docs)
        assert docs[0] == np.array([25])  # z

    def test_generator(fname, docs):
        for replace_lower_freq_word in [True, False]:
            dictionary = Dictionary(min_count=2, replace_lower_freq_word=replace_lower_freq_word)
            dictionary.fit_from_list(docs)
            docs_from_list = dictionary.transform_from_list(docs)
            for doc, generated_doc in zip(docs_from_list, dictionary.generator_transform_from_fname(fname)):
                np.testing.assert_array_equal(doc, generated_doc)

    create_test_corpus_files()

    for fname in [TEXT8_LIKE_FILE_NAME, DOC_FILE_NAME]:
        test_fit_from_fname(fname=fname)
        test_fit_with_replace_and_add_special_from_fname(fname=fname)

        if 'text8' in fname:
            continue

        with open(fname) as f:
            docs = f.readlines()
            test_fit_from_docs(docs)
            test_fit_min_count1(docs)
            test_fit_with_replace_and_add_special_from_fname(docs)
            test_transform(docs)
            test_generator(fname, docs)

    delete_test_corpus_files()
