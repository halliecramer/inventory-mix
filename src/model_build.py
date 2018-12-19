import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import psycopg2
import config
from datetime import datetime
from psycopg2.extras import execute_values
from sklearn.svm import SVC
import os
import sys

class Reservations(object):
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.con = psycopg2.connect(dbname=config.redshift['dbname'],
                                    host=config.redshift['host'],
                                    port=config.redshift['port'],
                                    user=config.redshift['user'],
                                    password=config.redshift['password'])
    def _get_data(self):
        '''Imports user data directly from redshift,
           formats, scales for scoring'''

        with open('sql/inventory_mix.sql','r') as f:
            sql = f.read()
        df = pd.read_sql(sql, self.con)
        target = ['is_reserved']
        num_cols = ['daily_spend', 'number_available_cars']
        cat_cols = ['region', 'day_of_week', 'month', 'model', 'model_year', 'alg_trim', 'color']
        date_cols = ['date_start']
        features = num_cols + cat_cols
        for var in cat_cols:
                label = LabelEncoder()
                df[var] = label.fit_transform(df[var].astype('str'))
        for var in date_cols:
            df[var] = df[var].notnull()
        self.X = df[features].fillna(0)
        self.y = df[target].values.ravel()

    def _split_data(self, oversample=1, scale=1):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            random_state=101,
                                                            test_size=0.35)
        y_train = y_train.ravel()
        if oversample:
            ros = SMOTE()
            X_train, y_train = ros.fit_sample(X_train, y_train)
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def fit_model(self, model='rf', oversample=True, scale=False, n_estimators=10,
                  info=True, roc=False, max_depth=25, learning_rate=0.1,
                  max_features='sqrt', min_samples_leaf=3):
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
            feature_df = pd.DataFrame([np.array(self.X.columns), self.model.feature_importances_]).T
            feature_df.columns = ['Feature','Value']
            print ("\nFEATURE IMPORTANCES")
            print(feature_df.sort_values('Value', ascending=False))
        if roc:
            self._plot_roc()

    def _plot_roc(self):
        probas_ = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, probas_)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(fpr, tpr, linestyle='--')
        plt.plot([0, 1], [0, 1], linestyle='--', color='k',label='Luck')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def make_predictions(self, calibrate_probas=1, p=0):
        self.fit_model(model='ab', info=True, roc=False, oversample=True, scale=True)
        if calibrate_probas:
            calibrated = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
            calibrated.fit(self.X, self.y)
            self.scores = calibrated.predict_proba(self.X)[:, 1]
        else:
            self.scores = self.model.predict_proba(self.X)[:, 1]
        self.y_pred = [a.item() for a in self.model.predict(self.X)]
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
            ax1.set_title('Calibration plots  (reliability curve)')
            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper right", ncol=2)
            plt.show()

if __name__ == '__main__':
    try:
        model = Reservations()
        # model.fit_model(model='rf', info=True, roc=True, oversample=True, scale=True)
        model.make_predictions(calibrate_probas=True, p=True)

    except IndexError:
        print('Expected: python model_build.py')
