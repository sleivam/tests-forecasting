import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder as ohe
import datetime as dt
from scipy.stats import poisson
import multiprocessing

class reader:
    """This class is mainly a placeholder for future updates in the data process"""
    def __init__(self, dt_source):
        self.data = pd.read_parquet(dt_source)

class model(reader):
    """This class is pretended to be used to fit the model and to predict the model"""
    def __init__(self, dt_source):
        super().__init__(dt_source)

    def make_features(self, df,
        categ_features = ['currency', 'listing_type', 'shipping_payment'],
        cont_features = ['date', 'current_price', 'minutes_active']):
        """Method to make de dataset to be used in the fit"""
        y = df[['sold_quantity']]
        X = df[['date', 'current_price', 'currency', 'listing_type', 'shipping_payment', 'minutes_active']]
        X_transf = pd.concat([X[cont_features], 
            pd.get_dummies(X[categ_features])], axis = 1)
        X_transf = (X_transf.
            assign(date = lambda X: pd.to_datetime(X.date).map(dt.datetime.toordinal)).
            assign(bias = 1))
        return({'y': y, 'X': X_transf})

    def fit_individual(self, features, model = sm.families.Poisson()):
        """Method to fit only one SKU or another grouping category"""
        y = features['y']
        X = features['X']
        model_ind = sm.GLM(y,
            X,
            family = model).fit()        
        return(model_ind)
    
    def make_future_data(self, base_data, days = 15):
        """Method to make data used to predict"""
        # Get las row from the base data 
        last_row_position = base_data.shape[0] - 1

        future_data = pd.DataFrame({'date': np.arange(np.max(base_data.date) + 1, np.max(base_data.date) + days + 1),
            'current_price': np.repeat(X_transf.current_price[last_row_position], days),
            'minutes_active': np.repeat(X_transf.minutes_active[last_row_position], days),
            'currency_REA': np.repeat(X_transf.currency_REA[last_row_position], days),
            'listing_type_classic': np.repeat(X_transf.listing_type_classic[last_row_position], days),
            'shipping_payment_free_shipping': np.repeat(X_transf.shipping_payment_free_shipping[last_row_position], days),
            'bias': np.repeat(X_transf.bias[last_row_position], days)
        })

        return(future_data)
        
    def predict_individual(self, sku, quantile = [0.05, 0.5, 0.95]):
        """Method to predict one SKU
        sku: SKU to be predicted
        quantile: a list with percentiles of prediction"""
        features = self.make_features(self.data.loc[self.data.sku == sku])
        future_data = self.make_future_data(features["X"])
        model_in = self.fit_individual(features)
        in_preds = {}
        in_preds["means"] = model_in.predict(future_data)
        # For now the variance is the same as mean, in future updates this should be 
        # generalized
        in_preds["variances"] = in_preds["means"]
        for i in quantile:
            in_preds["q_" + str(i)] = poisson.ppf(i, mu = in_preds['means'])
        return(pd.DataFrame(in_preds))


class mass_model(model):
    """This class contains the methods to fit and predict all sku in parallel"""
    def __init__(self, dt_source):
        super().__init__(dt_source)

    def fit_massive(self):
        pool = multiprocessing.Pool()



    




# ========== Pruebas de la clase ==========

data_path = "./tests-forecasting/train_data.parquet"

modelo = model(data_path)
feat = modelo.make_features(modelo.data.loc[modelo.data.sku == 464801])
fit_a = modelo.fit_individual(feat)
future = modelo.make_future_data(feat['X'])
modelo.predict_individual(464801)



a['means'] = fit_a.predict(future)
a['variances'] = a['means']
quantile = (0.05, 0.5, 0.95)

for i in quantile:
    a["q_" + str(quantile)] = poisson.ppf(i, mu = means)



a = {}
a['ola'] = 5
a
fit_a.predict(future)

modelo.fit_inidividual(464801)


df_prueba = df.loc[df.sku == 464801]

df_prueba[['sold_quantity']]


sm.GLM(np.array(y), np.array(X))


list_categ_features = ['currency', 'listing_type', 'shipping_payment']
list_cont_features = ['date', 'current_price', 'minutes_active']
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df['Date']=data_df['Date'].map(dt.datetime.toordinal)



X_transf = pd.concat([X[list_cont_features], 
    pd.get_dummies(X[list_categ_features])], axis = 1)
X_transf = (X_transf.
    assign(date = lambda X: pd.to_datetime(X.date).map(dt.datetime.toordinal)).
    assign(bias = 1))


model_ind = sm.GLM(np.array(y),
    X_transf,
    family = sm.families.Poisson()).fit()

model_ind.summary()


X_transf

# Expand date







