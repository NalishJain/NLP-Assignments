def laplace_smoothing_matrix(self):
    return (self.bigram_counts + 1) / (self.word_count_matrix[:, np.newaxis] + self.vocab_size)

def kneser_ney_smoothing_matrix(self, discount=0):
    discounted_probs = np.maximum(self.bigram_counts - discount, 0) / self.word_count_matrix[:, np.newaxis]
    alpha_word1 = (discount * np.sum(self.bigram_counts > 0, axis=1)) / self.word_count_matrix
    cont_probs = np.sum(self.bigram_counts > 0, axis=0) / np.sum(self.bigram_counts > 0)
    return discounted_probs + alpha_word1[:, np.newaxis] * cont_probs
