def generate_samples(self, emotion_id = 0, num_samples = 50):
        generated_samples = []
        for _ in range(num_samples):
            sample = self.generate_sample(emotion_id)
            generated_samples.append(sample)
        return generated_samples

def generate_sample(self, emotion_id = 0, max_length = 10):

    start_word = np.random.choice(['i', 'im', 'ive'], p = [0.7, 0.2, 0.1])
    current_word = start_word
    sample = [current_word]

    for _ in range(max_length - 1):

        current_word_index = self.vocabulary_index[current_word]
        probabilities = self.calculate_probability_emotion_row(current_word, emotion_id)

        if np.all(probabilities == 0):
            break

        probabilities /= probabilities.sum()

        next_word_index = np.random.choice(self.vocab_size, p = probabilities)
        next_word = self.index_vocabulary[next_word_index]

        sample.append(next_word)
        current_word = next_word

    return ' '.join(sample)

def calculate_probability_emotion_row(self, first_wrod, emotion_id=0):
    non_zero_indices = np.nonzero(self.bigram_counts[self.vocabulary_index[first_wrod], :])[0]
    emo_probs = np.zeros((self.vocab_size, 6))
    for second_word_index in non_zero_indices:
        k = self.emotion_scores(first_wrod + " " +  self.index_vocabulary[second_word_index])
        prob_score = []
        for label_score in range(6):
            prob_score.append(k[label_score]['score'])
        emo_probs[second_word_index] = np.array(prob_score)
    first_word_mat = self.bigram_counts / self.word_count_matrix[:, np.newaxis]
    return first_word_mat[self.vocabulary_index[first_wrod],:] + emo_probs[:, emotion_id]

def emotion_scores(self, sample):
    emotion=classifier(sample)
    return emotion[0]