from collections import defaultdict, Counter
import re

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merge_rules = []
        self.token_list=[]
        

    def merge_vocab(self, pair, v_in):
        # Convert the pair into a regular expression for matching
        pattern = re.compile(re.escape(' '.join(pair)))

        

        # Find the most frequent occurrence of the pair in the vocabulary
        max_freq_word = max(v_in, key=v_in.get)

        # Replace occurrences of the most frequent pair with a merged version in the vocabulary
        v_out = {pattern.sub(''.join(pair), word): freq for word, freq in v_in.items()}

        # Remove the individual symbols of the merged pair from the vocabulary
        v_out.pop(' '.join(pair), None)
        v_out.pop(pair[0], None)
        v_out.pop(pair[1], None)

        return v_out

    def get_stats(self, vocab):
        # Initialize a defaultdict to store counts of consecutive symbol pairs
        pairs_count = defaultdict(int)

        # Iterate over each word in the vocabulary
        for word, frequency in vocab.items():
            # Split each word into a list of symbols
            symbols = word.split()

            # Count the frequency of consecutive symbol pairs
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs_count[pair] += frequency

        return dict(pairs_count)

    def build_vocab(self, corpus, num_merges):
        # Tokenize the input corpus into words
        words = corpus.split()

        # Separate each character in each word by a space and append </w> to each word
        tokens = [' '.join(list(word) + ['$']) for word in words]
        # print(tokens)
        all_characters = ''.join([char for word in tokens for char in word if char != ' '])

        # Get unique characters
        self.token_list = list(set(all_characters))

        # Count the frequency of each token using Counter
        token_counts = Counter(tokens)

        # Return the vocabulary as a dictionary
        self.vocab = dict(token_counts)
        # print(self.vocab)

        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)  # Step 2

            if not pairs:
                break

            # Step 3: Find the best pair and merge
            best = max(pairs, key=pairs.get)
            self.token_list.append(best[0]+best[1])
            self.merge_rules.append(best)
            self.vocab = self.merge_vocab(best, self.vocab)

        return self.vocab, self.merge_rules,self.token_list
    
    def tokenize(self, input_text):
        # Tokenize the input text using the precomputed merge rules
       

        words = input_text.split()

        # Separate each character in each word by a space and append </w> to each word
        tokens = [' '.join(list(word) + ['$']) for word in words]

        # Apply precomputed merge rules
        for rule in self.merge_rules:
            tokens = [token.replace(' '.join(rule), ''.join(rule)) for token in tokens]
        
        tokens2=[]

        for i in tokens:
            tokens2.append(i.replace(" ",","))
        str2=""
        for i in range(0,len(tokens2)):
            if(i!=len(tokens2)-1):            
                str2+=tokens2[i]+","
            else:
                str2+=tokens2[i]

        return str2
        

file = open('D:\Downloads 2\corpus.txt', 'r')
corpus = file.read().replace('\n', '')

# Example usage:
# corpus = "old old old old old old old older older older finest finest finest finest finest finest finest finest finest lowest lowest lowest lowest"
num_merges = 100

tokenizer = Tokenizer()
vocabulary, merge_rules,tok = tokenizer.build_vocab(corpus, num_merges)

tokenized=tokenizer.tokenize("kanye made tailor swift famous")

# print("Vocabulary:", vocabulary)
print("Merge Rules:", merge_rules)
# print("Tokens:",tok)
print("Tokenised Version:",tokenized)
