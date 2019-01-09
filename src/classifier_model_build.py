import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
import random
from pandas.tseries.holiday import USFederalHolidayCalendar as hcal
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble.forest import _generate_unsampled_indices
import pickle
import psycopg2
import config
import itertools as itt
from datetime import datetime
from psycopg2.extras import execute_values
from sklearn.svm import SVC
import os
import sys

class Reservations(object):
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
        self.con = None

    def _get_data(self):
        '''Imports user data directly from redshift,
           formats, scales for scoring'''

        with open('sql/logit_reservations.sql','r') as f:
            sql = f.read()
        self.con = psycopg2.connect(dbname=config.redshift['dbname'],
                                    host=config.redshift['host'],
                                    port=config.redshift['port'],
                                    user=config.redshift['user'],
                                    password=config.redshift['password'])
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
        target = ['is_reserved']
        num_cols = ['rolling_avg_spend', 'number_available_cars', 'is_canvas_2_0',
                    # 'focus_count', 'fusion_count', 'fiesta_count', 'escape_count',
                    # 'c-max hybrid_count', 'explorer_count', 'edge_count', 'mustang_count',
                    'is_weekend', 'is_holiday', 'vehicle_fee', 'is_neutral_color',
                    'min_vehicle_fee', 'is_apr_thru_jul', 'number_available_model']
        cat_cols = ['region', 'model', 'model_year']
        date_cols = ['date_start']
        dummy_list = []
        for var in cat_cols:
            x = pd.get_dummies(df[var])
            dummy_list.append(list(x))
            df = pd.concat([df, x], axis=1)
        dummy_list = list(itt.chain.from_iterable(dummy_list))
        self.features = num_cols + dummy_list
        # self.features = num_cols + cat_cols
        # for var in cat_cols:
        #     label = LabelEncoder()
        #     df[var] = label.fit_transform(df[var].astype('str'))
        #     if self.pickle:
        #         with open("saved_model/{0}_labels.pkl".format(var), 'wb') as f:
        #             pickle.dump(label, f)
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
                  info=True, roc=False, reliability=False, max_depth=10, learning_rate=0.1,
                  max_features='sqrt', min_samples_leaf=3,
                  C=0.1, gamma=1, kernel='rbf', pickle=0, permutation_feat=False):
        self.pickle = pickle
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(oversample=oversample, scale=scale)
        if model == 'rf':
            m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       max_features=max_features, min_samples_leaf=min_samples_leaf)
        elif model == 'gb':
            m = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           max_features=max_features, min_samples_leaf=min_samples_leaf,
                                           learning_rate=learning_rate)
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
        if permutation_feat:
            imp = self.permutation_importances(self.model, self.X_train, self.y_train)
            features_df = pd.DataFrame(data={'Feature':np.array(self.X.columns), 'Value':imp}).set_index('Feature')
            print ("\nFEATURE IMPORTANCES")
            print(features_df.sort_values('Value', ascending=False))
        else:
            feature_df = pd.DataFrame([np.array(self.X.columns), self.model.feature_importances_]).T
            feature_df.columns = ['Feature','Value']
            print ("\nFEATURE IMPORTANCES")
            print(feature_df.sort_values('Value', ascending=False))
        if roc:
            self._plot_roc()
        if reliability:
            self._plot_reliability()
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

    def _plot_reliability(self, oversample=True, scale=False, bins=10, calibrate_probas=True):
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(oversample=oversample, scale=scale)
        classifiers = {"Logistic regression": LogisticRegression(solver='warn'),
                       "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_leaf=3),
                       "Gradient Boosted": GradientBoostingClassifier(n_estimators=100, max_depth=10, max_features='sqrt', min_samples_leaf=3),
                       "SVC": SVC(C=0.1, gamma=1, kernel='rbf')}
        reliability_scores = {}
        y_score = {}
        for method, clf in classifiers.items():
            model = clf.fit(self.X_train, self.y_train)
            if method == "SVC":
                # Use SVC scores (predict_proba returns already calibrated probabilities)
                y_score[method] = model.decision_function(self.X)
                reliability_scores[method] = calibration_curve(self.y, y_score[method], n_bins=bins, normalize=True)
            else:
                if calibrate_probas:
                    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                    calibrated.fit(self.X, self.y)
                    y_score[method] = calibrated.predict_proba(self.X)[:, 1]
                    reliability_scores[method] = calibration_curve(self.y, y_score[method], n_bins=bins, normalize=False)
                else:
                    y_score[method] = model.predict_proba(self.X)[:, 1]
                    reliability_scores[method] = calibration_curve(self.y, y_score[method], n_bins=bins, normalize=False)
        # Plot curves
        plt.figure(0, figsize=(8, 8))
        plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
        for method, (y_score_mean, empirical_prob) in reliability_scores.items():
            scores_not_nan = np.logical_not(np.isnan(empirical_prob))
            plt.plot(y_score_mean[scores_not_nan],
                     empirical_prob[scores_not_nan], label=method)
        plt.ylabel("Empirical probability")
        plt.legend(loc=0)

        plt.subplot2grid((3, 1), (2, 0))
        for method, _y_score in y_score.items():
            _y_score = (_y_score - _y_score.min()) / (_y_score.max() - _y_score.min())
            plt.hist(_y_score, range=(0, 1), bins=bins, label=method,
                     histtype="step", lw=2)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend(loc='upper center', ncol=2)
        plt.show()

    def grid_search(self, model='gb', oversample=False, scale=False, scoring='f1'):
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(oversample=oversample, scale=scale)
        if model == 'gb':
            m = GradientBoostingClassifier()
            param_grid = {"max_features" : ['sqrt'],
                          "n_estimators" : [100],
                          "max_depth" : [10, 15],
                          "min_samples_leaf" : [3, 5, 8],
                          "learning_rate" : [0.1]}
        elif model == 'rf':
            m = RandomForestClassifier()
            param_grid = {"max_features" : ['sqrt', 'auto'],
                          "n_estimators" : [100],
                          "max_depth" : [10, 25, 45],
                          "min_samples_leaf" : [3, 10]}
        elif model == 'ab':
            m = AdaBoostClassifier()
            param_grid = {"n_estimators" : [200, 250, 300, 350, 400],
                          "learning_rate" : [0.01, 0.1, 1]}
        CV = GridSearchCV(estimator=m, param_grid=param_grid, cv=3, verbose=10, scoring=scoring, n_jobs=-1)
        CV.fit(self.X_train, self.y_train)
        print("\nBest Params:")
        print(CV.best_params_)

    def pickle_model(self):
        with open("saved_model/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == '__main__':
    try:
        model = Reservations()
        model.fit_model(model='gb', info=True, roc=False, reliability=False, pickle=False, min_samples_leaf=3, max_depth=10)
        # model.grid_search(model='gb', oversample=True, scale=False, scoring='f1')

    except IndexError:
        print('Expected: python model_build.py')
