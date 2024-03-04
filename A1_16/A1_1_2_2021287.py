file = open('corpus.txt', 'r')
corpus = file.read().replace('\n', ' ')
print(corpus)
num_merges = 7

tokenizer = Tokenizer()
vocabulary, merge_rules,tok = tokenizer.build_vocab(corpus, num_merges)

tokenized=tokenizer.tokenize("kanye made tailor swift famous")


print("Vocabulary:", vocabulary)
print("Merge Rules:", merge_rules)
print("Tokens:",tok)
print("Tokenised Version:",tokenized)

f = open(  "merge_rules.txt", "w")
for i in merge_rules:
    f.write(i[0] + "," + i[1] + "\n")
f.close()

f = open( "tokens.txt", "w")
for i in tok:
    f.write(i + "\n")
f.close()

f = open("tokenized.txt", "w")
n = int(input("Enter the number of sentences you want to tokenize: "))
for i in range(n):
    s = input("Enter the sentence: ")
    f.write(tokenizer.tokenize(s) + "\n")
f.close()