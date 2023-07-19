import os
num_thread = 16
os.environ['OMP_NUM_THREADS'] = str(num_thread)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_thread)
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
# classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier    
from sklearn.svm import SVC

from . import config

class Model:
    def ReadData(self, path_data, return_df=False):
        df = pd.read_csv(path_data, index_col=0, dtype={'# idx':'i', 'type':'i'})
        df.index.names = ['idx']
        df['type'] = df['type'] // 10 # remove initial condition info
        df['type'] = df['type'].astype(str).replace(config.type_dict_r)
        
        X = df.drop(config.pm_list[1:], axis=1)
        y = df['type']
        
        if return_df == True:
            return df, X, y
        else:
            return X, y
 
    def Predict(self, path_train, path_test, mc, ratio=0.3, verbose='t'):
        mc_dict = {
            'rf':   RandomForestClassifier(random_state=config.random_state, n_jobs=num_thread),
            'xgb':  XGBClassifier(random_state=config.random_state, nthread=num_thread),
            'lgbm': LGBMClassifier(random_state=config.random_state, n_jobs=num_thread, objective='multiclass', num_class=len(config.type_dict)),
            'cat':  CatBoostClassifier(random_state=config.random_state, thread_count=num_thread, silent=True, allow_writing_files=False),
            'lr':   LogisticRegression(random_state=config.random_state, n_jobs=num_thread, solver='sag', max_iter=100000),
            'svm':  SVC(random_state=config.random_state, probability=True),
        }
        ohe = True if mc in ['xgb'] else False # one-hot encoding

        t0 = timer()

        if path_test == 'none':
            X, y = self.ReadData(path_train)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, stratify=y, random_state=config.random_state)
        else:
            X_train, y_train = self.ReadData(path_train)
            X_test,  y_test  = self.ReadData(path_test)
        
        if ohe: y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)

        mc_dict[mc].fit(X_train, y_train)

        y_pred  = mc_dict[mc].predict(X_test)
        # y_score = mc_dict[mc].predict_proba(X_test)

        if ohe:
            y_test, y_pred = y_test.values.argmax(axis=1), y_pred.argmax(axis=1)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        if verbose == 't':
            print()
            print('# Trainset : %s' % path_train, X_train.shape)
            print('# Testset  : %s' % path_test, X_test.shape)
            print('# Machine  : %s' % mc)
            print('# Accuracy : %f' % acc)
            print(cm)
            print()

        t1 = timer()
        print('Predict : %fs' % (t1-t0))
