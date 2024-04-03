from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

global precision, recall, fscore, accuracy

X = np.load("model/X.txt.npy")
Y = np.load("model/Y.txt.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]


with open('model/tfidf.txt', 'rb') as file:
    tfidf = pickle.load(file)
file.close()
X = tfidf.fit_transform(X).toarray()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

if os.path.exists('model/svm.txt'):
    with open('model/svm.txt', 'rb') as file:
        svm_cls = pickle.load(file)
    file.close()
else:
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)
    file.close()

if os.path.exists('model/lgbm.txt'):
    with open('model/lgbm.txt', 'rb') as file:
        lgbm_cls = pickle.load(file)
    file.close()
else:
    lgbm_cls = LGBMClassifier()
    lgbm_cls.fit(X_train, y_train)
    with open('model/lgbm.txt', 'wb') as file:
        pickle.dump(lgbm_cls, file)
    file.close()

with open('model/rf.txt', 'rb') as file:
    rf_cls = pickle.load(file)
file.close()    

def RunSVM(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        global X_train, X_test, y_train, y_test
        precision = []
        accuracy = []
        fscore = []
        recall = []
        predict = svm_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)
        output = ""
        output+='<tr><td><font size="" color="lime"><b>SVM With Feature Scaling</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(accuracy[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(precision[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(recall[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(fscore[0])+'</b></td>'

        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("SVM with Feature Scaling Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)     

def RunLGBM(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        global X_train, X_test, y_train, y_test
        
        predict = lgbm_cls.predict(X_test)
        acc = accuracy_score(y_test,predict)*100
        p = precision_score(y_test,predict,average='macro') * 100
        r = recall_score(y_test,predict,average='macro') * 100
        f = f1_score(y_test,predict,average='macro') * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(acc)
        output = ""
        output+='<tr><td><font size="" color="lime"><b>SVM with Feature Scaling</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(accuracy[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(precision[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(recall[0])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(fscore[0])+'</b></td>'

        output+='<tr><td><font size="" color="lime"><b>Light GBM</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(accuracy[1])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(precision[1])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(recall[1])+'</b></td>'
        output+='<td><font size="" color="aqua"><b>'+str(fscore[1])+'</b></td>'
        
        LABELS = ['Normal URL','Phishing URL']
        conf_matrix = confusion_matrix(y_test, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("LGBM Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()    
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)    



def getData(arr):
    data = ""
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 0:
            data += arr[i]+" "
    return data.strip()        

def PredictAction(request):
    if request.method == 'POST':
        global rf_cls, tfidf
        url_input = request.POST.get('t1', False)
        test = []
        arr = url_input.split("/")
        if len(arr) > 0:
            data = getData(arr)
            print(data)
            test.append(data)
            test = tfidf.transform(test).toarray()
            print(test)
            print(test.shape)
            predict = rf_cls.predict(test)
            print(predict)
            predict = predict[0]
            output = ""
            if predict == 0:
               output = "<font color=\"lightblue\"><u><b>It's Safe to Browse</b></u>: " + url_input + "</font>"


            if predict == 1:
                output = "<font color=\"red\" ><u><b>Phishing Detected in this URL</b></u>: " + url_input + "</font>"
            context = {'data': output}
            return render(request, 'Predict.html', context)
        else:
            # Return a default response when URL input is not valid
            output = "Entered URL is not valid"
            context = {'data': output}
            return render(request, 'Predict.html', context)
    
    # Return a default response if the request method is not POST
    return HttpResponse("Invalid request method")


# views.py

from django.shortcuts import render
from django.http import HttpResponse

# Existing imports

# Existing views
def welcome_view(request):
    return render(request, 'welcome.html')

def index(request):
    return render(request, 'index.html', {})

def Predict(request):
    return render(request, 'Predict.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "Hemanth" and password == "Hemanth":
            context= {'data':'Welcome '+user}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'AdminLogin.html', context)

# New view for Streamlit app
def streamlit_view(request):
    # Define the URL where your Streamlit app is running
    streamlit_url = "http://localhost:8501"  # Change this to match your Streamlit server's URL
    
    return render(request, 'streamlit_view.html', {'streamlit_url': streamlit_url})
