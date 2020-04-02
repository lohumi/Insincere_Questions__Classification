import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict 

from nltk.corpus import stopwords
from collections import defaultdict
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import eli5


#print os.listdir('./Quora-Insincere-Questions-Classifica
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Train data: \nRows:{} \ncolumns: {}' .format(train.shape[0],train.shape[1]) )
print(train.columns)


print('Test data: \nRows:{} \ncolumns: {}' .format(test.shape[0],test.shape[1]) )
print(test.columns)

#check for the number of positive and negative classes
pd.crosstab(index=train.target,columns='count')
#col_0     count
#target         
#0       1225312
#1         80810


#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    #define additional stop words that are not contained in the dictionary
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)
    #Generate the word cloud
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    #set the plot parameters
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
# Select insincere questions from training set
insincere = train.loc[train['target']==1]
plot_wordcloud(insincere['question_text'],title='Word Cloud of Insincere Questions')

#Select sincere questions from training dataset
sincere = train.loc[train['target']==0]
plot_wordcloud(sincere['question_text'],title='Word Cloud of sincere Questions')

#side by side plot comparison using N-gram
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, col, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df[col]:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df

#Function to construct side by side comparison plots
def comparison_plot(df_1,df_2,col_1,col_2, space):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")

    ax[0].set_xlabel('Word count', size=14)
    ax[0].set_ylabel('Words', size=14)
    ax[0].set_title('Top words in sincere questions', size=18)

    ax[1].set_xlabel('Word count', size=14)
    ax[1].set_ylabel('Words', size=14)
    ax[1].set_title('Top words in insincere questions', size=18)

    fig.subplots_adjust(wspace=space)
    
    plt.show()
    
#Obtain sincere and insincere ngram based on 1 gram (top 20)
sincere_1gram = generate_ngrtrain["question_text"]ams(train[train["target"]==0], 'question_text', 1, 20)
insincere_1gram = generate_ngrams(train[train["target"]==1], 'question_text', 1, 20)
#compare the bar plots
comparison_plot(sincere_1gram,insincere_1gram,'word','wordcount', 0.25)

#Obtain sincere and insincere ngram based on 2 gram (top 20)
sincere_1gram = generate_ngrams(train[train["target"]==0], 'question_text', 2, 20)
insincere_1gram = generate_ngrams(train[train["target"]==1], 'question_text', 2, 20)
#compare the bar plots
comparison_plot(sincere_1gram,insincere_1gram,'word','wordcount', 0.25)


#Obtain sincere and insincere ngram based on 3 gram (top 20)
sincere_1gram = generate_ngrams(train[train["target"]==0], 'question_text', 3, 20)
insincere_1gram = generate_ngrams(train[train["target"]==1], 'question_text', 3, 20)
#compare the bar plots
comparison_plot(sincere_1gram,insincere_1gram,'word','wordcount', 0.25)

# Number of words in the questions
#Insincere questions have more words per question
train['word_count']= train["question_text"].apply(lambda x:len(str(x).split()))

test['word_count']= test["question_text"].apply(lambda x:len(str(x).split()))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x='word_count',y='target',data=train,ax=ax,palette=sns.color_palette("RdYlGn_r", 10),orient='h')
ax.set_xlabel('Word Count', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Word Count distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()                           
              

# Number of characters in the questions
# Insincere questions have more characters than sincere questions
train["char_length"] = train["question_text"].apply(lambda x: len(str(x)))
test["char_length"] = test["question_text"].apply(lambda x: len(str(x)))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="char_length", y="target", data=train, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Character Length', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Character Length distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()

# Number of stop words in the questions
# Insincere questions have more stop words than sincere questions
train["stop_words_count"] = train["question_text"].apply(lambda x:len([ w for w in str(x).lower().split() if w in STOPWORDS ]))
test["stop_words_count"] = test["question_text"].apply(lambda x:len([ w for w in str(x).lower().split() if w in STOPWORDS ]))
fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="stop_words_count", y="target", data=train, ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Number of stop words', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Number of Stop Words distribution', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()

# Mean word length in the questions
train["word_length"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["word_length"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

fig, ax = plt.subplots(figsize=(15,2))
sns.boxplot(x="word_length", y="target", data=train[train['word_length']<train['word_length'].quantile(.99)], ax=ax, palette=sns.color_palette("RdYlGn_r", 10), orient='h')
ax.set_xlabel('Mean word length', size=10, color="#0D47A1")
ax.set_ylabel('Target', size=10, color="#0D47A1")
ax.set_title('[Horizontal Box Plot] Distribution of mean word length', size=12, color="#0D47A1")
plt.gca().xaxis.grid(True)
plt.show()


# Get the tfidf vectors
tfidf_vec = TfidfVectorizer(stop_words='english',ngram_range=(1,3))
tfidf_vec.fit_transform(train['question_text'].values.tolist() + test['question_text'].values.tolist())

train_tfidf= tfidf_vec.transform(train['question_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test['question_text'].values.tolist())
y_train = train["target"].values
x_train = train_tfidf
x_test = test_tfidf

model = linear_model.LogisticRegression(C=5., solver='sag')
model.fit(x_train, y_train)
y_test = model.predict_proba(x_test)[:,1]


eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')





