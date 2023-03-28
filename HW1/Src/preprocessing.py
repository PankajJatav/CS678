from collections import defaultdict
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class Preprocess:
    def __init__(self, data_exs, emb_dim=300, glove_path=None):
        self.data = data_exs
        self.vocab = []
        self.emb_dim = emb_dim
        self.embedding = [np.random.rand(self.emb_dim), np.random.rand(self.emb_dim)]
        self.word_dict = defaultdict(int)
        self.glove_path = glove_path
        self.preprocess()

    def preprocess(self):
        self.process_vocab()
        self.process_embedding()

    def make_dict(self):
        for line in self.data:
            for word in line.words:
                self.word_dict[word] += 1

    def make_vocab_from_dict(self, min_count = 1):
        for key in self.word_dict:
            if self.word_dict[key] >= min_count:
                self.vocab += [key]

    def remove_stop_word(self):
        for word in self.vocab:
            if word not in stop_words:
                self.vocab.remove(word)

    def process_vocab(self):
        if not len(self.data):
            return
        self.make_dict()
        self.make_vocab_from_dict(min_count=2)
        # self.remove_stop_word()

    def process_embedding(self):
        if not self.glove_path:
            self.embedding = np.random.rand(len(self.vocab)+2, self.emb_dim)
        else:
            em_vocab, embeddings = [], []
            with open(self.glove_path, 'rt') as embeddings_file:
                data = embeddings_file.read().strip().split('\n')
            for i in range(len(data)):
                current_word = data[i].split(' ')[0]
                if current_word in self.vocab:
                    current_embeddings = [float(val) for val in data[i].split(' ')[1:]]
                    em_vocab.append(current_word)
                    embeddings.append(current_embeddings)

            for word in self.vocab:
                if word not in em_vocab:
                    self.embedding.append(np.random.rand(self.emb_dim))
                else:
                    ind = em_vocab.index(word)
                    self.embedding.append(embeddings[ind])