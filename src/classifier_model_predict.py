import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from pandas.tseries.holiday import USFederalHolidayCalendar as hcal
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import pickle
import psycopg2
import config
import itertools as itt
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
        self.con = psycopg2.connect(dbname=config.redshift['dbname'],
                                    host=config.redshift['host'],
                                    port=config.redshift['port'],
                                    user=config.redshift['user'],
                                    password=config.redshift['password'])

    def _import_model(self):
        with open("saved_model/model.pkl", 'rb') as f:
            self.model = pickle.load(f)

    def _get_data(self, scale=0):
        '''Imports user data directly from redshift,
           formats, scales for scoring'''

        with open('sql/logit_reservations.sql','r') as f:
            sql = f.read()
        df = pd.read_sql(sql, self.con)
        self.con.close()
        # model_list = ['Focus','Fusion','Fiesta','Escape','C-Max Hybrid','Explorer','Edge','Mustang']
        # for var in model_list:
        #     daily_counts = df[df['model']==var].groupby(['date_start','region'])['vin'].count().reset_index()
        #     daily_counts = pd.DataFrame(daily_counts)
        #     daily_counts.columns = ['date_start', 'region', (str(var) + '_count').lower()]
        #     df = pd.merge(df, daily_counts, how='left', on=['date_start','region'])
        rolling_avg_spend = pd.DataFrame(df.groupby(['date_start','region'])['daily_spend'].mean().rolling(window=21, min_periods=1).mean()).reset_index()
        rolling_avg_spend.columns = ['date_start','region','rolling_avg_spend']
        df = pd.merge(df, rolling_avg_spend, how='left', on=['date_start','region'])
        holidays = hcal().holidays(start='2017-05-05', end='2023-1-01')
        df['is_holiday'] = df['date_start'].apply(lambda x: 1 if x in holidays else 0)
        df['year'] = df['date_start'].apply(lambda x: x.year)
        df['season'] = df['month'].apply(lambda x: 'winter' if np.isin(x, ['Dec','Jan','Feb']) == True
                                            else ('spring' if np.isin(x, ['Mar','Apr','May']) == True
                                            else ('summer' if np.isin(x, ['Jun','Jul','Aug']) == True
                                            else 'fall')))
        df['is_apr_thru_jul'] = df['month'].apply(lambda x: 1 if x in ['Apr','May','Jun','Jul'] else 0)
        df['is_neutral_color'] = df['color'].apply(lambda x: 1 if x in ['White','Grey','Silver','Black'] else 0)
        self.orig = df.copy()
        target = ['is_reserved']
        num_cols = ['rolling_avg_spend', 'number_available_cars', 'is_canvas_2_0',
                    # 'focus_count', 'fusion_count', 'fiesta_count', 'escape_count',
                    # 'c-max hybrid_count', 'explorer_count', 'edge_count', 'mustang_count',
                    'is_weekend', 'is_holiday', 'vehicle_fee', 'is_neutral_color',
                    'min_vehicle_fee', 'is_apr_thru_jul']
        cat_cols = ['region', 'model', 'model_year']
        other_cols = ['date_start', 'alg_trim', 'make']
        dummy_list = []
        for var in cat_cols:
            x = pd.get_dummies(df[var])
            dummy_list.append(list(x))
            df = pd.concat([df, x], axis=1)
        dummy_list = list(itt.chain.from_iterable(dummy_list))
        self.features = num_cols + cat_cols + other_cols
        model_features = num_cols + dummy_list
        # self.features = num_cols + cat_cols
        # for var in cat_cols:
        #     with open("saved_model/{0}_labels.pkl".format(var), 'rb') as f:
        #         label = pickle.load(f)
        #     df[var] = label.transform(df[var].astype('str'))
        # for var in date_cols:
        #     df[var] = df[var].notnull()
        self.y = df[target].values.ravel()
        if scale:
            with open("saved_model/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            self.X = scaler.transform(self.X)
        else:
            self.X = df[model_features].fillna(0)

    def make_predictions(self, calibrate_probas=1, p=1, scale=0, print_csv=True):
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
            if print_csv:
                groupby_cols = ['model', 'model_year', 'alg_trim']
                unique_months = inventory_scores['month'].unique()
                unique_cities = inventory_scores['region'].unique()
                with open('output.csv', 'w') as f:
                    for y in unique_cities:
                        print("\n{} Canvas 2.0 Best and Worst Performing Cars by Month".format(y), file=f)
                        partition_scores = inventory_scores[inventory_scores['region']==y]
                        for i in unique_months:
                            performance = partition_scores[(partition_scores['month']==i) & (partition_scores['is_canvas_2_0']==1) & (partition_scores['make']!='Lincoln')].groupby(groupby_cols)['proba_reserved'].agg(['mean'])
                            print("\n5 Highest Probabilities in {}".format(i), file=f)
                            print(performance.sort_values(by='mean',ascending=False).head(5), file=f)
                            print("\n5 Lowest Probabilities in {}".format(i), file=f)
                            print(performance.sort_values(by='mean',ascending=True).head(5), file=f)
            else:
                groupby_cols = ['region', 'month', 'model', 'model_year', 'alg_trim', 'is_neutral_color']
                performance = inventory_scores[(inventory_scores['is_canvas_2_0']==1) & (inventory_scores['make']!='Lincoln')].groupby(groupby_cols)['proba_reserved'].agg(['mean'])
                print("\n10 Highest Probabilities: ")
                print(performance.sort_values(by='mean',ascending=False).head(10))
                print("\n10 Lowest Probabilities: ")
                print(performance.sort_values(by='mean',ascending=True).head(10))
            # reliability diagram
            fop_uncalibrated, mpv_uncalibrated = calibration_curve(self.y, self.model.predict_proba(self.X)[:,1], n_bins=10, normalize=True)
            fop_calibrated, mpv_calibrated = calibration_curve(self.y, self.scores, n_bins=10, normalize=True)
            # plot perfectly calibrated
            fig = plt.figure(1, figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.plot([0, 1], [0, 1], linestyle='--')
            # plot calibrated reliability
            ax1.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', color='blue', label='uncalibrated')
            ax1.plot(mpv_calibrated, fop_calibrated, marker='.', color='green', label='calibrated')
            ax2.hist(self.model.predict_proba(self.X)[:,1], range=(0, 1), bins=10, histtype="step", lw=2, color='blue', label='uncalibrated')
            ax2.hist(self.scores, range=(0, 1), bins=10, histtype="step", lw=2, color='green', label='calibrated')

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            # ax1.legend(loc="lower right")
            ax1.set_title('Calibration plots  (reliability curve)')
            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper right", ncol=2)
            plt.show()

        # def output(self):
        #     '''Appends data to a redshift table'''
        #     insert_query = "insert into analytics_pipeline.traffic_score (model_version, score_timestamp, merged_id, score, prediction) values %s"
        #     # scores = scores[:10]
        #     model_version = ['model_{0}_day_activity_window'.format(self.activity_window)] * len(self.scores)
        #     score_date = [datetime.now()] * len(self.scores)
        #     new_rows = list(zip(model_version, score_date, self.users, self.scores, self.y_pred))
        #     cur = self.con.cursor()
        #     execute_values(cur, insert_query, new_rows)
        #     self.con.commit()
        #     self.con.close()

if __name__ == '__main__':
    try:
        model = PredictScores()
        model.make_predictions(calibrate_probas=True, p=True, scale=False)

    except IndexError:
        print('Expected: python model_predict.py')
