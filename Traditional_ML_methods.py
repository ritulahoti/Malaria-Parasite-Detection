import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

### Dataset were used from kaggle (https://www.kaggle.com/c/malaria-parasite-detection/data)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from os import listdir
from os.path import isfile, join
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from skimage import util
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage import io
from skimage.io import imread, imshow


def feature_glcm(img_arr):
    gCoMat = greycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')    
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')   
    feat_glcm = np.array([contrast[0][0],dissimilarity[0][0],homogeneity[0][0],energy[0][0],correlation[0][0]])

    return (feat_glcm)


def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def feature_lbp(img_arr):
    feat_lbp = local_binary_pattern(img_arr,5,2,'uniform')#.reshape(img.size[0]*img.size[1])
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist= np.array(lbp_hist,dtype=float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lbp_prob = replaceZeroes(lbp_prob)
    lbp_energy = np.nansum(lbp_prob*2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))  
    feat_lbp = np.array([lbp_energy, lbp_entropy])
    
    return (feat_lbp)


def feature_gabor(img_arr):
    gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
    gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilt,8)
    gabor_hist = np.array(gabor_hist,dtype=float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob**2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    feat_gabor = np.array([gabor_energy,gabor_entropy])
    
    return (feat_gabor)


# Feature Extraction

df_para = pd.DataFrame(columns=['Name', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation','lbp_energy','lbp_entropy','gabor_energy','gabor_entropy', 'Label'])
df_un = pd.DataFrame(columns=['Name', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation','lbp_energy','lbp_entropy','gabor_energy','gabor_entropy', 'Label'])

def load_dataset(path, classes):
    class_images = []    
    for cls in classes :
        #cls_imgs = []
        img_names = os.listdir(path + cls + "/")
        if cls == "Parasitized":
            df_para['Name'] = img_names
            i = 0
            for original_name in img_names :
                
                ### Morphological Operation
                
                '''gray = cv2.cvtColor(cv2.imread(path + cls + "/" + original_name), cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
                # closing operation 
                kernel = np.ones((3, 3), np.uint8) 
                closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2) 
                # Background area using Dialation 
                bg = cv2.dilate(closing, kernel, iterations = 2)'''
                
                ### K-means Clustering
                img = cv2.cvtColor(cv2.imread(path + cls + "/" +  original_name), cv2.COLOR_BGR2RGB)
                vectorized = img.reshape((-1,3))
                vectorized = np.float32(vectorized)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 3
                attempts=10
                ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                result_image = res.reshape((img.shape))
                
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

                np_img = np.array(result_image)
                f1, f2, f3, f4, f5 = feature_glcm(np_img)
                f6, f7 = feature_lbp(np_img)
                f8, f9 = feature_gabor(np_img)
                df_para.loc[i,'Contrast'] = f1
                df_para.loc[i,'Dissimilarity'] = f2
                df_para.loc[i,'Homogeneity'] = f3
                df_para.loc[i,'Energy'] = f4
                df_para.loc[i,'Correlation'] = f5
                df_para.loc[i,'lbp_energy'] = f6
                df_para.loc[i,'lbp_entropy'] = f7
                df_para.loc[i,'gabor_energy'] = f8
                df_para.loc[i,'gabor_entropy'] = f9
                df_para.loc[pd.IndexSlice[i,'Label']] = "1"
                #df_para['label'] = "1"
                i=i+1
        else:
            #df_un['label'] = df_un['label'].fillna(0)
            #df_un['label'] = "0"
            df_un['Name'] = img_names
            i = 0
            for original_name in img_names :
                
                ### Morphological Operation
                
                '''gray = cv2.cvtColor(cv2.imread(path + cls + "/" + original_name), cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
                # closing operation 
                kernel = np.ones((3, 3), np.uint8) 
                closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations = 2) 
                # Background area using Dialation 
                bg = cv2.dilate(closing, kernel, iterations = 2)'''
                
                ### K-means Clustering
                img = cv2.cvtColor(cv2.imread(path + cls + "/" +  original_name), cv2.COLOR_BGR2RGB)
                vectorized = img.reshape((-1,3))
                vectorized = np.float32(vectorized)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 2
                attempts=10
                ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                result_image = res.reshape((img.shape))
                
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

                np_img = np.array(result_image)
                f1, f2, f3, f4, f5 = feature_glcm(np_img)
                f6, f7 = feature_lbp(np_img)
                f8, f9 = feature_gabor(np_img)
                df_un.loc[i,'Contrast'] = f1
                df_un.loc[i,'Dissimilarity'] = f2
                df_un.loc[i,'Homogeneity'] = f3
                df_un.loc[i,'Energy'] = f4
                df_un.loc[i,'Correlation'] = f5
                df_un.loc[i,'lbp_energy'] = f6
                df_un.loc[i,'lbp_entropy'] = f7
                df_un.loc[i,'gabor_energy'] = f8
                df_un.loc[i,'gabor_entropy'] = f9
                df_un.loc[pd.IndexSlice[i,'Label']] = "0"
                #df_un['label'] = "0"
                i=i+1

    return df_para, df_un

df_para, df_un = load_dataset("../input/detectparasite/Parasite/train/", ["Parasitized", "Uninfected"])

df_final = df_para.append(df_un, ignore_index=True)
df_final

df_final.to_csv('trainKmeansSegment.csv', index="False")

df_final = pd.read_csv("/kaggle/input/kmeansseg/trainKmeansSegment.csv")

df_final


corr = df_final.corr()
corr.style.background_gradient(cmap='coolwarm')


X = df_final.drop(['Name','Contrast','Dissimilarity','Correlation','lbp_energy','Label'],axis=1)
y = df_final['Label']


# Splitting the dataset into training and validation sets.75% for training and 25% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
type(X_test)

import pickle
from sklearn.externals import joblib

#Import svm model
from sklearn import svm
#Create a svm Classifier
SVM = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
SVM.fit(X, y)

# Validation using SVM

#Import svm model
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

#Create a logistic regression model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(solver='lbfgs')
LR.fit(X, y)

# Validation using Logistic Regression

#Create a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
NB = GaussianNB()
# Train the model using the training sets
NB.fit(X,y)

# Validation using Naive Bayes

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
NB = GaussianNB()
# Train the model using the training sets
NB.fit(X_train,y_train)

NB.score(X_test, y_test)


#Import Random forest model
from sklearn.ensemble import RandomForestClassifier

RF= RandomForestClassifier(random_state=1)
RF.fit(X, y)


# Validation using Random Forest

#Import Random forest model
from sklearn.ensemble import RandomForestClassifier

modelRF= RandomForestClassifier(random_state=1)
modelRF.fit(X_train, y_train)

modelRF.score(X_test, y_test)

joblib.dump([NB, SVM, LR, RF], 'Traditional_models.pkl')


with open ('Traditional_models','wb') as f:
    pickle.dump([NB, SVM, LR, RF],f)

with open ('Traditional_models','rb') as f:
    model=pickle.load(f)


#0:NB, 1:SVM, 2:LR, 3:RF
model[0].score(X_test, y_test)


df_test = pd.DataFrame(columns=['Name', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation','lbp_energy','lbp_entropy','gabor_energy','gabor_entropy', 'Label'])

def load_dataset1(path):
    img_names = os.listdir(path)
    df_test['Name'] = img_names
    i = 0
    for original_name in img_names :
        '''img = cv2.cvtColor(cv2.imread(path + original_name), cv2.COLOR_BGR2RGB)
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))'''
        
        gray = cv2.cvtColor(cv2.imread(path + original_name),cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        np_img = np.array(sure_bg)
        
        #result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

        #np_img = np.array(result_image)
        f1, f2, f3, f4, f5 = feature_glcm(np_img)
        f6, f7 = feature_lbp(np_img)
        f8, f9 = feature_gabor(np_img)
        df_test.loc[i,'Contrast'] = f1
        df_test.loc[i,'Dissimilarity'] = f2
        df_test.loc[i,'Homogeneity'] = f3
        df_test.loc[i,'Energy'] = f4
        df_test.loc[i,'Correlation'] = f5
        df_test.loc[i,'lbp_energy'] = f6
        df_test.loc[i,'lbp_entropy'] = f7
        df_test.loc[i,'gabor_energy'] = f8
        df_test.loc[i,'gabor_entropy'] = f9
        i=i+1
        
    return df_test


df_test_SVM = load_dataset1("../input/detectparasite/Parasite/test/")


X_test_SVM = df_test_SVM.drop(['Name','Contrast','Dissimilarity','Correlation','lbp_energy','Label'],axis=1)
# Predicted output SVM for test dataset
y_pred = clf.predict(X_test_SVM)
y_pred

df_test1 = pd.DataFrame(y_pred, columns=['Label'])
df_test_SVM['Label'] = df_test1

X_test_SVM = df_test_SVM.drop(['Homogeneity', 'Energy','lbp_entropy','gabor_energy','gabor_entropy'],axis=1)
X_test_SVM


X_test_SVM.to_csv('testKmeansSegment.csv', index="False")


f1 = pd.read_csv("../input/submissions/Submission1.csv")
f1.drop('Label', axis=1, inplace=True)
f2 = pd.read_csv("testKmeansSegment.csv")
f1 = f1.merge(f2, left_on='Name', right_on='Name', how='outer')
f1.to_csv('testKmeansSegmentSorted.csv', index=False)

