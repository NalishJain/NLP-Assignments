from collections import defaultdict, Counter
import re

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merge_rules = []
        self.token_list=[]

    def merge_vocab(self, pair, vocab):
        pattern = re.compile(re.escape(' '.join(pair)))
        max_freq_word = max(vocab, key=vocab.get)
        vocab2 = {pattern.sub(''.join(pair), word): freq for word, freq in vocab.items()}
        vocab2.pop(' '.join(pair), None)
        vocab2.pop(pair[0], None)
        vocab2.pop(pair[1], None)
        return vocab2
    
    def get_stats(self, vocab):
        pairs_count = defaultdict(int)
        for word, frequency in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs_count[pair] += frequency
        return dict(pairs_count)
    
    def build_vocab(self, corpus, num_merges):
        words = corpus.split()
        tokens = [' '.join(list(word) + ['$']) for word in words]
        all_characters = ''.join([char for word in tokens for char in word if char != ' '])
        self.token_list = list(set(all_characters))
        token_counts = Counter(tokens)
        self.vocab = dict(token_counts)
        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)  # Step 2
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.token_list.append(best[0]+best[1])
            self.merge_rules.append(best)
            self.vocab = self.merge_vocab(best, self.vocab)
        return self.vocab, self.merge_rules,self.token_list
    
    def tokenize(self, input_text):       
        words = input_text.split()
        tokens = [' '.join(list(word) + ['$']) for word in words]
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
    
#################################################################
    
