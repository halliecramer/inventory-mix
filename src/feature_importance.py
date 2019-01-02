import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble.forest import _generate_unsampled_indices
import pickle
import psycopg2
import config
from datetime import datetime
from psycopg2.extras import execute_values
from sklearn.svm import SVC
import os
import sys

class PermutationImportance(object):
    def __init__(self):
        self.model = None
        self.pickle = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.features = None

    def _get_data(self):
        '''Imports user data directly from redshift,
           formats, scales for scoring'''

        with open('sql/inventory_mix.sql','r') as f:
            sql = f.read()
        con = psycopg2.connect(dbname=config.redshift['dbname'],
                                host=config.redshift['host'],
                                port=config.redshift['port'],
                                user=config.redshift['user'],
                                password=config.redshift['password'])
        df = pd.read_sql(sql, con)
        con.close()
        # model_list = ['Focus','Fusion','Fiesta','Escape','C-Max Hybrid','Explorer','Edge','Mustang']
        # for var in model_list:
        #     daily_counts = df[df['model']==var].groupby(['date_start','region'])['vin'].count().reset_index()
        #     daily_counts = pd.DataFrame(daily_counts)
        #     daily_counts.columns = ['date_start', 'region', (str(var) + '_count').lower()]
        #     df = pd.merge(df, daily_counts, how='left', on=['date_start','region'])
        rolling_avg_spend = pd.DataFrame(df.groupby(['date_start','region'])['daily_spend'].mean().rolling(window=21, min_periods=1).mean()).reset_index()
        rolling_avg_spend.columns = ['date_start','region','rolling_avg_spend']
        df = pd.merge(df, rolling_avg_spend, how='left', on=['date_start','region'])
        df['year'] = df['date_start'].apply(lambda x: x.year)
        target = ['is_reserved']
        num_cols = ['rolling_avg_spend',
                    # 'focus_count', 'fusion_count', 'fiesta_count', 'escape_count',
                    # 'c-max hybrid_count', 'explorer_count', 'edge_count', 'mustang_count',
                    'number_available_cars']
        cat_cols = ['region', 'day_of_week', 'month', 'model', 'model_year', 'alg_trim', 'color', 'year']
        date_cols = ['date_start']
        self.features = num_cols + cat_cols
        for var in cat_cols:
                label = LabelEncoder()
                df[var] = label.fit_transform(df[var].astype('str'))
                if self.pickle:
                    with open("saved_model/{0}_labels.pkl".format(var), 'wb') as f:
                        pickle.dump(label, f)
        for var in date_cols:
            df[var] = df[var].notnull()
        self.X = df[self.features].fillna(0)
        self.y = df[target].values.ravel()

    def _split_data(self, oversample=1, scale=1):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            random_state=101,
                                                            test_size=0.25)
        y_train = y_train.ravel()
        if oversample:
            ros = SMOTE()
            X_train, y_train = ros.fit_sample(X_train, y_train)
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            if self.pickle:
                with open("saved_model/scaler.pkl", 'wb') as f:
                    pickle.dump(scaler, f)
        return X_train, X_test, y_train, y_test

    def fit_model(self, model='rf', oversample=True, scale=False, n_estimators=100,
                  info=True, max_depth=45, learning_rate=0.1,
                  max_features='auto', min_samples_leaf=10,
                  C=0.1, gamma=1, kernel='rbf', pickle=0):
        self.pickle = pickle
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(oversample=oversample, scale=scale)
        if model == 'rf':
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       max_features=max_features, min_samples_leaf=min_samples_leaf)
        elif model == 'gb':
            m = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           max_features=max_features, min_samples_leaf=min_samples_leaf)
        elif model == 'ab':
            m = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        elif model == 'svc':
            m = SVC(C=c, gamma=gamma, kernel=kernel)
        self.model = m.fit(self.X_train, self.y_train)
        if info:
            self.y_pred = self.model.predict(self.X_test,)
            description = """
                \nModel: {}, oversample: {}, scale: {}, \nn_estimators: {},
                \nmax_depth: {}, \nlearning_rate: {},
                \nmax_features: {}, \nmin_samples_leaf: {}"""
            print(description.format(model, oversample, scale, n_estimators,
                          max_depth, learning_rate,
                          max_features, min_samples_leaf))
            print("\nMETRICS")
            print("Model recall: {}".format(recall_score(self.y_test, self.y_pred)))
            print("Model precision: {}".format(precision_score(self.y_test, self.y_pred)))
            print("Model accuracy: {}".format(self.model.score(self.X_test, self.y_test)))
            print("Model f1 score: {}".format(f1_score(self.y_test, self.y_pred)))
            print ("\nCONFUSION MATRIX")
            print (confusion_matrix(self.y_test, self.y_pred))
            print ("\nkey:")
            print (" TN   FP ")
            print (" FN   TP ")
            imp = self.permutation_importances(self.model, self.X_train, self.y_train)
            features_df = pd.DataFrame(data={'Feature':np.array(self.X.columns), 'Importance':imp}).set_index('Feature')
            print ("\nFEATURE IMPORTANCES")
            print(features_df.sort_values('Importance', ascending=False))
        if self.pickle:
            self.pickle_model()

    def permutation_importances(self, rf, X_train, y_train):
        baseline = self.oob_classifier_accuracy(rf=self.model, X=self.X_train, y=self.y_train)
        imp = []
        for col in range(X_train.shape[1]):
            save = X_train[:, col].copy()
            X_train[:, col] = np.random.permutation(X_train[:, col])
            n = self.oob_classifier_accuracy(self.model, self.X_train, self.y_train)
            X_train[:, col] = save
            imp.append(baseline - n)
        return np.array(imp)

    def oob_classifier_accuracy(self, rf, X, y):
        n_samples = len(X)
        n_classes = len(np.unique(y))
        predictions = np.zeros((n_samples, n_classes))
        for tree in rf.estimators_:
            unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
            tree_preds = tree.predict_proba(X[unsampled_indices, :])
            predictions[unsampled_indices] += tree_preds

        predicted_class_indexes = np.argmax(predictions, axis=1)
        predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]

        oob_score = np.mean(y == predicted_classes)
        return oob_score

if __name__ == '__main__':
    try:
        model = PermutationImportance()
        model.fit_model(model='rf', info=True, oversample=True, scale=False)

    except IndexError:
        print('Expected: python model_build.py')
