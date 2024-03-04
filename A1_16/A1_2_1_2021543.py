class BigramLM_efficient:
    def __init__(self):
        self.vocab_size = 0
        self.vocabulary_index = {}
        self.word_count = {}
        self.index_vocabulary = {}
        self.bigram_counts = None
        self.bigram_probabilities = None
        self.dataset = None


    def build_corpus(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip().split() for line in file]
        self.dataset =  sentences

    def build_vocab(self):
        for line in self.dataset:
            for word in line:
                if word not in self.vocabulary_index:
                    self.vocabulary_index[word] = self.vocab_size
                    self.index_vocabulary[self.vocab_size] = word
                    self.word_count[word] = 0
                    self.vocab_size += 1
                self.word_count[word] += 1

    def build_probability_matrix(self, mode, discount=0, emotion_id=0):
        self.bigram_probabilities = np.zeros((self.vocab_size, self.vocab_size), dtype=float)

        if mode == 0:
            self.bigram_probabilities = self.calculate_probability_matrix()
        elif mode == 1:
            self.bigram_probabilities = self.laplace_smoothing_matrix()
        elif mode == 2:
            self.bigram_probabilities = self.kneser_ney_smoothing_matrix(discount=discount)


    def calculate_probability_matrix(self):
        return self.bigram_counts / self.word_count_matrix[:, np.newaxis]

    def learn(self, file_path):
        self.build_corpus(file_path)
        self.build_vocab()

        self.bigram_counts = np.zeros((self.vocab_size, self.vocab_size), dtype=int)
        self.word_count_matrix = np.array(list(self.word_count.values()))

        for line in self.dataset:
            for index in range(len(line) - 1):
                first_word_index = self.vocabulary_index[line[index]]
                second_word_index = self.vocabulary_index[line[index + 1]]
                self.bigram_counts[first_word_index, second_word_index] += 1