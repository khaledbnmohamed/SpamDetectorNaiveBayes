import pandas as pd
import string
import pprint
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



with open('SMSSpamCollection.csv') as f:
    df = pd.read_table(f, sep='\t', header=None, names=["label",'sms_message'],
                          lineterminator='\n')


df["label"] = df.label.map({"ham": 0, "spam": 1})

# df['label'] = map(df,di)

#############count vector from scratch on small  example data list##########


documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']


lower_case_documents = []
sans_punctuation_documents = []
frequency_list = []
frequency_list = Counter()
for element in documents:
    	lower_case_documents.append(element.lower())

for i in lower_case_documents:
	sans_punctuation_documents.append(i.translate(str.maketrans('','',string.punctuation)))
for i in lower_case_documents:
	frequency_list.update(Counter(i.split()))  ##not completely correct needs to append all results
#pprint.pprint(frequency_list) 
#print(sans_punctuation_documents)

# print(df.head(5))
# print(table[1:6,])    



count_vector= CountVectorizer(input=documents, encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64)
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray() #matrix creation then change to array

# print(doc_array)

frequency_matrix = pd.DataFrame({'Column1':doc_array[:,0],'Column2':doc_array[:,1]})

# print(frequency_matrix)


############## Done scratching & back to real world##################

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test) #WE DON'T DIT OUR TEST DATA HERE



#FINISHED ^^^^ DATA PREPROCESSING AND NEXT SECTION IS IMPLEMENTING THE TRAINING ALGORITHM####

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)

# print(predictions)

print('Accuracy score: ', format(accuracy_score(y_test, predictions) * 100) + "%")
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

