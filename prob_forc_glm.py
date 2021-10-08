import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder as ohe
import datetime as dt

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

    def fit_inidividual(self, features, model = sm.families.Poisson()):
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

        future_data = pd.DataFrame({'date': np.arange(np.max(base_data.date) + 1, np.max(base_data.date) + 15 + 1),
            'current_price': np.repeat(X_transf.current_price[last_row_position], 15),
            'minutes_active': np.repeat(X_transf.minutes_active[last_row_position], 15),
            'currency_REA': np.repeat(X_transf.currency_REA[last_row_position], 15),
            'listing_type_classic': np.repeat(X_transf.listing_type_classic[last_row_position], 15),
            'shipping_payment_free_shipping': np.repeat(X_transf.shipping_payment_free_shipping[last_row_position], 15),
            'bias': np.repeat(X_transf.bias[last_row_position], 15)
        })

        return(future_data)
        
    def predict_individual(self, sku):
        features = self.make_features(self.data.loc[self.data.sku == sku])
        future_data = self.make_future_data(features['X'])
        model_in = self.fit_individual(features)
        ind_preds = model_in.predict(future_data)

        return(ind_preds)



data_path = "./tests-forecasting/train_data.parquet"

modelo = model(data_path)
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







np.array(X_transf)

encod.n_features_in_
encod.categories_
modelo
