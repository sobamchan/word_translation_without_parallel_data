

class Dictionary(object):

    def __init__(self, i2w, w2i, lang):
        assert len(i2w) == len(w2i)
        self.i2w = i2w
        self.w2i = w2i
        self.lang = lang
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.i2w)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.i2w[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.w2i

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.i2w) != len(y):
            return False
        return self.lang == y.lang and\
            all(self.i2w[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.i2w) == len(self.w2i)
        for i in range(len(self.i2w)):
            assert self.w2i[self.i2w[i]] == i

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.w2i[word]

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        self.i2w = {k: v for k, v in self.i2w.items() if k < max_vocab}
        self.w2i = {v: k for k, v in self.i2w.items()}
        self.check_valid()
