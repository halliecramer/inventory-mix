import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import pickle
import psycopg2
import config
from datetime import datetime
from psycopg2.extras import execute_values
import os
import sys
import bisect

class PredictScores(object):
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.y_pred = None
        self.orig = None
        self.features = None

    def _import_model(self):
        with open("saved_model/model.pkl", 'rb') as f:
            self.model = pickle.load(f)

    def _get_data(self, scale=0):
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
        self.orig = df.copy()
        target = ['is_reserved']
        num_cols = ['rolling_avg_spend',
                    # 'focus_count', 'fusion_count', 'fiesta_count', 'escape_count',
                    # 'c-max hybrid_count', 'explorer_count', 'edge_count', 'mustang_count',
                    'number_available_cars']
        cat_cols = ['region', 'day_of_week', 'month', 'model', 'model_year', 'alg_trim', 'color', 'year']
        date_cols = ['date_start']
        self.features = num_cols + cat_cols
        for var in cat_cols:
            with open("saved_model/{0}_labels.pkl".format(var), 'rb') as f:
                label = pickle.load(f)
            # label_classes = label.classes_.tolist()
            # df[var] = df[var].map(lambda s: '<unknown>' if s not in label_classes else s)
            # bisect.insort_left(label_classes, '<unknown>')
            # label.classes_ = label_classes
            df[var] = label.transform(df[var].astype('str'))
        for var in date_cols:
            df[var] = df[var].notnull()
        self.y = df[target].values.ravel()
        if scale:
            with open("saved_model/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            self.X = scaler.transform(self.X)
        else:
            self.X = df[self.features].fillna(0)

    def make_predictions(self, calibrate_probas=1, p=1, scale=0):
        self._import_model()
        self._get_data()
        if calibrate_probas:
            calibrated = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
            calibrated.fit(self.X, self.y)
            self.scores = calibrated.predict_proba(self.X)[:, 1]
        else:
            self.scores = self.model.predict_proba(self.X)[:, 1]
        self.y_pred = self.model.predict(self.X)
        if p == 1:
            print("\nMETRICS")
            print("Model recall: {}".format(recall_score(self.y, self.y_pred)))
            print("Model precision: {}".format(precision_score(self.y, self.y_pred)))
            print("Model accuracy: {}".format(self.model.score(self.X, self.y)))
            print("Model f1 score: {}".format(f1_score(self.y, self.y_pred)))
            print ("\nCONFUSION MATRIX")
            print (confusion_matrix(self.y, self.y_pred))
            print ("\nkey:")
            print (" TN   FP ")
            print (" FN   TP ")
            inventory_scores = pd.concat([self.orig[self.features], pd.Series(self.y), pd.Series(self.y_pred), pd.Series(self.scores)], axis=1)
            inventory_scores.columns = self.features + ['is_reserved', 'predict_reserved', 'proba_reserved']
            groupby_cols = ['region', 'model', 'model_year', 'alg_trim', 'color']
            performance = inventory_scores.groupby(groupby_cols)['proba_reserved'].agg(['mean'])
            print("\n20 Best performing cars: ")
            print(performance.sort_values(by='mean',ascending=False).head(20))
            print("\n20 Worst performing cars: ")
            print(performance.sort_values(by='mean',ascending=True).head(20))

if __name__ == '__main__':
    try:
        model = PredictScores()
        model.make_predictions(calibrate_probas=True, p=True, scale=False)

    except IndexError:
        print('Expected: python model_predict.py')
