emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

for emotion in range(len(emotions)):
    generated_samples = bigram_model.generate_samples(num_samples = 50, emotion_id=emotion)
    output_file = f'gen_{emotions[emotion]}.txt'

    with open(output_file, 'w', encoding='utf-8') as file:
        for sample in generated_samples:
            file.write(sample + '\n')

corpus_path = 'corpus.txt'
labels_path = 'labels.txt'

with open(corpus_path, 'r', encoding='utf-8') as file:
    texts = [line.strip() for line in file]

with open(labels_path, 'r', encoding='utf-8') as file:
    labels = [line.strip() for line in file]


X_train, y_train = texts, labels

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
testing_samples = []

X_test = []
y_test = []

for emotion in emotions:
    emotion_file_path = f'Generated files/gen_{emotion}_filtered.txt'
    with open(emotion_file_path, 'r', encoding='utf-8') as file:
        emotion_samples = [line.strip() for line in file]

        X_test.extend(emotion_samples)
        y_test.extend([emotion] * len(emotion_samples))

y_test = list(y_test)

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

svc_model = SVC(kernel = 'linear', C = 120, gamma = 0.002, break_ties = True, probability = True)
svc_model.fit(X_train_tfidf, y_train)

y_pred = svc_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

param_grid = {
    'C': [100, 115, 125],
    'gamma': [0.001, 0.007, 0.0096],
    'kernel': ['linear', 'rbf']
}

svc_model = SVC(break_ties = True, probability = True)

grid_search = GridSearchCV(estimator = svc_model, param_grid = param_grid, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_grid = best_model.predict(X_test_tfidf)

accuracy_grid = accuracy_score(y_test, y_pred_grid)
classification_rep = classification_report(y_test, y_pred_grid)

print("Best Parameters:", best_params)
print("Accuracy with Grid Search:", accuracy_grid * 100)
print("Classification Report:\n", classification_rep)