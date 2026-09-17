import numpy as np
import pandas
text_metki = pandas.read_csv('klassiffied_mail_text.csv')

#print(text_metki.head())

def process_email(text):
    text = text.lower()
    return list(set(text.split()))

text_metki ['words'] = text_metki['text'].apply(process_email)

#print(text_metki.columns)
#print(text_metki.head())

model={}


for index,row in text_metki.iterrows():
    for word in row['words']:
        
        if word not in model :
            model[word]={'spam': 1, 'ham': 1}
        if word in model:
            if row['spam']==1:
                model[word]['spam']+=1
            else:
                model[word]['ham']+=1
    
def predict_naive_bayes(text):
    total=len(text_metki)
    num_spam=sum(text_metki['spam'])
    num_ham=sum(text_metki['spam']==0)

    text=set(text.lower().split())

    spam_p=[1.0]
    ham_p=[1.0]

    for word in text:
        if word in model:
            spam_p.append(model[word]['spam']/num_spam)
            ham_p.append(model[word]['ham']/num_ham)
    zn_spam=np.prod(spam_p)* num_spam/total
    zn_ham=np.prod(ham_p)* num_ham/total

    ocenka = zn_spam/ (zn_spam+zn_ham)
    return ocenka
    
