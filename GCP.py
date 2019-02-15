
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
# Get data

data = []
data = pd.read_csv('~/Downloads/train.csv', header = 0, sep = ',', encoding = 'latin-1', error_bad_lines = False)
data = data.loc[:,('Sentiment', 'SentimentText')]
data = data[0:20000]


# CLEAN DATA
##Function to strip Links, removes HTTP/HTTPS:
def strip_links(text):
    import re
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ' ')
    return text

#function to strip all @,# and entities (&,^,<,>, etc) but keeps the "'" (They're)
def strip_all_entities(text):
    import string
    entity_prefixes = ['@','#', "'"]
    for separator in  string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

## Function to removing stop words
# Should this be done? will stop words not provide order to the sentence and will that order not matter when fed to the first layer of the NN?
# Can define your own stopwords if needed
def remove_stop_words(text):
    from nltk.corpus import stopwords
    stop_words = {'amp', 'quot'}
    #stop_words = set(stopwords.words('english'))
    words = text_to_word_sequence(text)
    newwords = [word for word in words if word not in stop_words]
    text = ' '.join(newwords)
    return text

## Fucntion to remove any string/tweet that has non-alpha numeric characters in a pandas dataframe
def remove_non_alphanum(text):
    for i in range(len(text)):
        try:
            text[i].encode(encoding = 'utf-8').decode('ascii')
        except UnicodeDecodeError:
            text = text.drop(i, inplace= False)
    text = text.reset_index(drop = True)
    return text

## Clean Tweets
data['CleanedText'] = ""
for i in range(len(data)):
    data.loc[i, 'CleanedText'] = remove_stop_words(strip_all_entities(strip_links(data.loc[i, 'SentimentText'])))

data['CleanedText'] = remove_non_alphanum(data['CleanedText'])

# Lemitization
## Should this be done? effects?

lemmatizer = WordNetLemmatizer()
for i in range(len(data)):
    words1 = text_to_word_sequence(data.loc[i, 'CleanedText'])
    newwords1 = []
    for j in words1:
        j = lemmatizer.lemmatize(j)
        newwords1.append(j)
    data.loc[i, 'CleanedText'] = ' '.join(newwords1)


# Instantiates a client
client = language.LanguageServiceClient()


# The text to analyze
document = {}
for i in range(len(data)):
    document[i] = types.Document(
    content = data.loc[i, 'CleanedText'],
    type=enums.Document.Type.PLAIN_TEXT)


# Detects the sentiment of the text
data["PredictedSentiment"] = " "
for i in range(len(document)):
    try:
        data.loc[i, "PredictedSentiment"] = client.analyze_sentiment(document=document[i]).document_sentiment.score
    except:
        pass

#Changing Sentiment to Binary Value for Comparison
data["PredictedBinarySentiment"] = " "
for i in range(len(document)):
    try:
        if data.loc[i, 'PredictedSentiment'] > 0.1:
            data.loc[i, 'PredictedBinarySentiment'] = 4
        elif data.loc[i, 'PredictedSentiment'] < 0.1:
            data.loc[i, 'PredictedBinarySentiment'] = 0
        else:
            data.loc[i, 'PredictedBinarySentiment'] = ' '
    except:
        pass


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
data['PredictedBinarySentiment'].replace(' ', np.nan, inplace=True)
data.dropna(subset=['PredictedBinarySentiment'], inplace=True)
confusion_matrix(data.Sentiment, data.PredictedBinarySentiment)
cr = classification_report(data.Sentiment, data.PredictedBinarySentiment)

data.groupby('PredictedBinarySentiment').count()

df = data.loc[:,('CleanedText','Sentiment')]
for i in range(len(df['CleanedText'])):
    if df.loc[i,'Sentiment'] == 4:
        df.loc[i,'Sentiment'] = 'positive'
    else:
        df.loc[i,'Sentiment'] = 'negative'


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(test_size=0.2)
x = data.loc[:,'CleanedText']
y = data.loc[:,'Sentiment']
for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


x_train, x_test, y_train, y_test = train_test_split(data.loc[:,'CleanedText'], data.loc[:,'Sentiment'], test_size= 0.20)

train = {'Sentiment': y_train, 'CleanedText': x_train}
train = pd.DataFrame(train)
train.groupby('Sentiment').count()
test = {'Sentiment': y_test, 'CleanedText': x_test}
test = pd.DataFrame(test)
test.groupby('Sentiment').count()

train = train[train.duplicated('CleanedText')]

df[:80000].to_csv('data.csv', header = False, index = False)


