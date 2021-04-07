from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import re
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import category_encoders as ce
from category_encoders import TargetEncoder


infile = open('pickle/vectorizer.pkl','rb')
vectorizer = pickle.load(infile)

infile = open('pickle/encoder_director_name.pkl','rb')
encoder_director_name = pickle.load(infile)

infile = open('pickle/encoder_lead_actor.pkl','rb')
encoder_lead_actor = pickle.load(infile)

infile = open('pickle/encoder_producer_name.pkl','rb')
encoder_producer_name = pickle.load(infile)

infile = open('pickle/encoder_production_company_1.pkl','rb')
encoder_production_company_1 = pickle.load(infile)

infile = open('pickle/encoder_sup_actor.pkl','rb')
encoder_sup_actor = pickle.load(infile)

infile = open('pickle/final_model.pkl','rb')
final_model = pickle.load(infile)


infile = open('pickle/standard_scaler.pkl','rb')
s_c = pickle.load(infile)

app = Flask(__name__)

@app.route('/') 
def home() : 
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict() : 
    test = pd.DataFrame(columns = ['belongs_to_collection','budget','runtime','Action','Adventure',
                                    'Animation','Comedy', 'Crime', 'Documentary','Drama','Family','Fantasy','Foreign','History','Horror','Music','Mystery',
                                    'Romance','Science Fiction','TV Movie','Thriller','War','Western',
                                    'production_company_1','Lead_Actor','Supporting_Actor','Director','Producer'])

    if request.form['collection'] == 'Yes' : 
        test.loc[0,'belongs_to_collection'] = 1 
    else : 
        test.loc[0,'belongs_to_collection'] = 0

    
    test.loc[0,'budget'] = int(request.form['budget'])
    test.loc[0,'runtime'] = int(request.form['runtime'])
    
    genre = request.form['genre']
    for i in test.columns[3:23] : 
        if genre == i : 
            test.loc[0,i] = 1
        else : 
            test.loc[0,i] = 0


    test.loc[0,'production_company_1'] = request.form['prod_company']
    test.loc[0,'Lead_Actor'] = request.form['lead_actor']
    test.loc[0,'Supporting_Actor'] = request.form['sup_actor']
    test.loc[0,'Director'] = request.form['director']
    test.loc[0,'Producer'] = request.form['producer']

    
    
    test.loc[:,'production_company_1'] = encoder_production_company_1.transform(test.loc[:,'production_company_1'])
    test.loc[:,'Lead_Actor'] = encoder_lead_actor.transform(test.loc[:,'Lead_Actor'])
    test.loc[:,'Supporting_Actor'] = encoder_sup_actor.transform(test.loc[:,'Supporting_Actor'])
    test.loc[:,'Director'] = encoder_director_name.transform(test.loc[:,'Director'])
    test.loc[:,'Producer'] = encoder_producer_name.transform(test.loc[:,'Producer'])

    comment = request.form['overview']
    comment = re.sub(r'[?|!|\'|"|#]',r'', comment)
    comment = re.sub(r'[.|,|)|(|\|/]',r' ', comment)
    comment = comment.replace("\n"," ")
    comment = re.sub('[^a-zA-Z]',' ', comment)
    comment = comment.lower()
    comment = comment.split()
    ps = SnowballStemmer(language='english')
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    data = [comment]
    ovrvw = vectorizer.transform(data).toarray()
    X_test = np.hstack((test,ovrvw))
    # Use sklearn version scikit-learn==0.22.2.post1
    X_test = s_c.transform(X_test)

    rating_pred = final_model.predict(X_test)

    return render_template('index.html',
                    rating='{}'.format(round(rating_pred[0],1)))

app.run(debug= True)
