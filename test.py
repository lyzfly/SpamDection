#
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from pandas.core.frame import  DataFrame
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class spam:

    def data(x_data,y_data):
        X_list = x_data
        y_list = y_data
        X = DataFrame(X_list)
        y = DataFrame(y_list)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21)
        print(len(y_train))
        print(len(y_test))
        return X_train,X_test,y_train,y_test

    def preprocessing(data_path):
        x_data=[]
        df = pd.read_csv(data_path,encoding='ISO-8859-1')
        df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
        df.columns=['label','text']
        text = df['text'].str.lower()
        label = df['label']
        for i in range(len(text)):
            token = RegexpTokenizer('[a-zA-Z]+').tokenize(text[i])
            filtered = [w for w in token if not w in stopwords.words('english')]
            x_data.append(filtered)
        return x_data,label

    '''for i in range(len(text)):
        message = text[i]
        list = 
        #disease_List = nltk.word_tokenize(message)
        
        print(filtered)'''

    def vetcorlize(x_train,x_test):
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(x_train)
        vectorizer.transform(x_test)
        return x_train,x_test


    def bayes(x_train,y_train,x_test):
        clf = MultinomialNB.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        return y_pred

    def test(self):
        hello = tf.constant("hello")
        sess = tf.compat.v1.Session()

    from sklearn.feature_extraction.text import TfidfVectorizer
    if __name__ == '__main__':
        x_data = preprocessing('spam.csv')[0]
        y_data = preprocessing('spam.csv')[1]
        X_train = data(x_data,y_data)[0]
        X_test = data(x_data,y_data)[1]
        y_train = data(x_data,y_data)[2]
        print(y_train)
        x_train = vetcorlize(X_train,X_test)[0]
        x_test = vetcorlize(X_train,X_test)[1]
        print(bayes(x_train,y_train,x_test))
