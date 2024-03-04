corpus_path = 'corpus.txt'
bigram_model = BigramLM_efficient()
bigram_model.learn(corpus_path)

bigram_model.build_probability_matrix(0)
top_bigrams = bigram_model.find_top_bigrams()
print("Top 5 Bigrams (Before smoothing):")
print()
for bigram, prob in top_bigrams:
    print(f"Bigram: '{bigram}',   Probability: {prob:.4f}")

bigram_model.build_probability_matrix(1)
top_bigrams = bigram_model.find_top_bigrams()
print("Top 5 Bigrams (After Laplace smoothing):")
print()
for bigram, prob in top_bigrams:
    print(f"Bigram: '{bigram}',   Probability: {prob:.4f}")

bigram_model.build_probability_matrix(2, 0.5)
top_bigrams = bigram_model.find_top_bigrams()
print("Top 5 Bigrams (After Kneser Ney smoothing):")
print()
for bigram, prob in top_bigrams:
    print(f"Bigram: '{bigram}',   Probability: {prob:.4f}")