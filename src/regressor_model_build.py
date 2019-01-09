import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import itertools as itt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import psycopg2
import config
from datetime import datetime
from sklearn.svm import SVR
import os
import sys

class Reservations(object):
    def __init__(self):
        self.model = None
        self.date = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.score = None
        self.wmape = None
        self.errors = None
        self.con = psycopg2.connect(dbname=config.redshift['dbname'],
                                    host=config.redshift['host'],
                                    port=config.redshift['port'],
                                    user=config.redshift['user'],
                                    password=config.redshift['password'])

    def _get_data(self):
        '''Imports user data directly from redshift,
           formats, scales for modelling'''

        with open('sql/count_reservations.sql','r') as f:
            sql = f.read()
        df = pd.read_sql(sql, self.con)

        df = df[df['cars_available'] > 0]
        df['proportion_reserved'] = df['reservations'] / df['cars_available']
        df['rolling_avg_spend'] = df['daily_spend'].rolling(10, min_periods=1).mean()
        df['season'] = df['month'].apply(lambda x: 'winter' if np.isin(x, ['Dec','Jan','Feb']) == True
                                            else ('spring' if np.isin(x, ['Mar','Apr','May']) == True
                                            else ('summer' if np.isin(x, ['Jun','Jul','Aug']) == True
                                            else 'fall')))
        listvar = ['focus','fusion','escape','explorer','edge','mustang',
                   'cmax_hybrid','fiesta','other',
                   'sedan','suv','hatchback','wagon','sports_car','pickup_truck',
                   'my_2015','my_2016','my_2017']
        for var in listvar:
            new_prop = df[var] / df['cars_available']
            new_prop = pd.DataFrame(new_prop, columns=['prop_' + str(var)])
            df = pd.merge(df, new_prop, left_index=True, right_index=True)

        target = ['proportion_reserved']
        num_cols = ['cars_available', 'rolling_avg_spend', 'prop_sedan', 'prop_suv',
                    'prop_hatchback', 'prop_wagon', 'prop_sports_car', 'prop_pickup_truck',
                    'prop_my_2015', 'prop_my_2016', 'prop_my_2017']
        cat_cols = ['region', 'season', 'day_of_week']
        dummy_list = []
        for var in cat_cols:
            x = pd.get_dummies(df[var])
            dummy_list.append(list(x))
            df = pd.concat([df, x], axis=1)
        dummy_list = list(itt.chain.from_iterable(dummy_list))
        features = num_cols + dummy_list
        self.date = df['date_start']
        self.X = df[features].fillna(0)
        self.y = df[target].values.ravel()

    def _split_data(self, scale=1):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            random_state=101,
                                                            test_size=0.3)
        y_train = y_train.ravel()
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def fit_model(self, model='rf', scale=False, n_estimators=100, info=True,
                  plot=False, max_depth=70, criterion='mse',
                  max_features='auto', min_samples_leaf=30, min_samples_split=5):
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(scale=scale)
        if model == 'rf':
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                       max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        elif model == 'gb':
            m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                           max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        elif model == 'ab':
            m = AdaBoostRegressor(n_estimators=n_estimators)
        self.model = m.fit(self.X_train, self.y_train)
        if info:
            self.y_pred = self.model.predict(self.X_test)
            self.errors = abs(self.y_pred - self.y_test)
            self.wmape = ((100 / len(self.y_test)) * (np.sum(self.errors) / np.sum(self.y_test)))
            self.score = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            description = """
                \nModel: {}, scale: {}, \nn_estimators: {},
                \nmax_depth: {}, \ncriterion: {},
                \nmax_features: {}, \nmin_samples_leaf: {},
                \nmin_samples_split: {}"""
            print(description.format(model, scale, n_estimators,
                          max_depth, criterion,
                          max_features, min_samples_leaf,
                          min_samples_split))
            print("\nMETRICS")
            print("Model Accuracy: {}%".format(100 - self.wmape))
            print("Model mean cross validation score: {}".format(np.mean(self.score)))
            print("Model MAE: {}".format(np.mean(self.errors)))
            print("Model RMSE: {}".format(np.sqrt(mean_squared_error(self.y_test, self.y_pred))))
            feature_df = pd.DataFrame([np.array(self.X.columns), self.model.feature_importances_]).T
            feature_df.columns = ['Feature','Value']
            print ("\nFEATURE IMPORTANCES")
            print(feature_df.sort_values('Value', ascending=False))
        if plot:
            self._plot_preds()

    def _plot_preds(self):
        true_data = pd.DataFrame(data = {'date': self.date, 'actual': self.y})
        predictions_data = pd.DataFrame(data = {'date': self.date, 'prediction': self.model.predict(self.X)})
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
        ax.plot(predictions_data['date'], predictions_data['prediction'], 'ro', markersize=3, label = 'prediction')
        plt.xticks(rotation = '60')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Proportion Reserved")
        plt.title("Actual and Predicted Values")
        plt.legend(loc="best")
        plt.show()

    def grid_search(self, model='rf', oversample=False, scale=False):
        self._get_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(scale=scale)
        if model == 'gb':
            m = GradientBoostingRegressor()
            param_grid = {"max_features" : ['auto', 'sqrt', 'log2'],
                          "n_estimators" : [10, 50, 100],
                          "max_depth" : [30, 40, 50, 60, 70],
                          "min_samples_leaf" : [30, 40, 50, 100],
                          "min_samples_split" : [2, 5, 10]}
        elif model == 'rf':
            m = RandomForestRegressor()
            param_grid = {"max_features" : ['auto', 'sqrt', 'log2'],
                          "n_estimators" : [10, 50, 100],
                          "max_depth" : [30, 40, 50, 60, 70],
                          "min_samples_leaf" : [30, 40, 50, 100],
                          "min_samples_split" : [2, 5, 10]}
        elif model == 'ab':
            m = AdaBoostRegressor()
            param_grid = {"n_estimators" : [200, 250, 300, 350, 400]}
        CV = GridSearchCV(estimator=m, param_grid=param_grid, cv=5, n_jobs=-1)
        CV.fit(self.X_train, self.y_train)
        print("\nBest Params:")
        print(CV.best_params_)

if __name__ == '__main__':
    try:
        model = Reservations()
        model.fit_model(model='rf', info=True, plot=True, scale=False)
        # model.grid_search(model='rf', oversample=False, scale=False)

    except IndexError:
        print('Expected: python model_build2.py')
