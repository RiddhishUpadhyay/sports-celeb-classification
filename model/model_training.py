import data_cleaning as dc

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix

xtrain, xtest, ytrain, ytest = train_test_split(dc.x,dc.y,random_state=0)
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svc', SVC(kernel = 'rbf',C = 10))
# ])
# pipe.fit(xtrain,ytrain)
# print(pipe.score(xtest, ytest))
# print(classification_report(ytest,pipe.predict(xtest)))

# use gridsearchcv to find best

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores = []
best_estimators = {}

import pandas as pd

for algo,mp in model_params.items():
    pipe = make_pipeline(StandardScaler(),mp['model'])
    clf = GridSearchCV(pipe,mp['params'],cv=5,return_train_score=False)
    clf.fit(xtrain, ytrain)
    scores.append({
        'model' : algo,
        'best_score' : clf.best_score_,
        'best_params' : clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(df)
# print(best_estimators) 
# print(best_estimators['svm'].score(xtest,ytest))

# we will use svm 
best_clf = best_estimators['logistic_regression']

import joblib
joblib.dump(best_clf, "saved_model.pkl")

import json
with open("class_dictionary.json",'w') as f:
    f.write(json.dumps(dc.class_dict))

ypred = best_clf.predict(xtest)
cm = confusion_matrix(ytest,ypred)
# print(cm)
print(best_clf.score(xtest,ytest))