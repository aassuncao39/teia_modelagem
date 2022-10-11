#warnings
import warnings
import platform
#time
import time
# data manipulation
import json
import numpy as np
import pandas as pd

# geospatial viz
import folium
from folium import Choropleth, Circle, Marker, Icon, Popup, FeatureGroup
from folium.plugins import HeatMap, FastMarkerCluster, MiniMap
from folium.features import GeoJsonPopup, GeoJsonTooltip

# plot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

#geospatial analysis
# import ee
import collections
import geopandas as gpd


#date manipulation
from datetime import timedelta
from datetime import datetime


#windows folder manipulation
import os

#statistics
import statistics
from statistics import mean
import scipy
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.stats import zscore

#regressions, metrics and models optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split #split database in training and testing data
from sklearn.ensemble import RandomForestRegressor #randomforest regression
from sklearn.linear_model import LinearRegression, Lasso #linear regression or multiple linear regression
from sklearn.neural_network import MLPRegressor #neural network
from sklearn.svm import SVR #svm regression
from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedKFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, make_scorer
from xgboost import XGBRegressor

#dataframe storage
import pickle

import collections

# ee.Authenticate()

collections.Callable = collections.abc.Callable
#ee.Authenticate()
# ee.Initialize()
# print('rolou')

"""
Criacao dos modelos
"""

warnings.simplefilter('ignore')

class Model():
    '''
        Model class encapsulate all methods to originate modelling results.
        
        Parameters:
        ------------
        X: pd.dataframe
            dataframe with independent (univariate) or independents (multivariate) variables
        y: pd.series
            dependent variable
        gv: bool
            set grid search to True or False
        model: Sklearn.model
            model desired | default = LinearRegression()
        wqp_unit: str
            enter wqp unit ex:mg/L
        mult_v: bool
            define if model is multivariate (more than one independent variable) or not. True or False
        m_type: str
            define the type of transformation to be applied in the data | default: linear (data as it is)
            Other types: 'poly', 'exp', 'log', 'standard_scaler', 'max_min'
        verbose: bool
            decide between printing processing results or not | default = True
        '''
    
    def __init__(self, X, y, gv=True, model=LinearRegression(), wqp_unit={}, mult_v = True, m_type = 'linear', verbose = True):
        self.X_name = ", ".join(list(X.columns))
        self.y_name = y.name
        self.gv = gv
        self.y = y
        self.model = model
        self.wqp_unit = wqp_unit
        self.result_dict = {}
        self.type = m_type
        self.mult_v = mult_v
        self.verbose = verbose
        if self.mult_v == False:
            self.X = np.array(X).reshape(-1,1)
            
        else:
            self.X = X

    def r2_func(self, y_obs, y_pred):
        ''' R2 metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        r2: float
            r2 metric
        '''
        
        if self.type == 'exp':
            y_obs, y_pred = np.exp(y_obs), np.exp(y_pred)
        else:
            pass
        y_hat = (1/len(y_obs))*sum(y_obs)
        TSS = sum((y_obs-y_hat)**2)
        RSS = sum((y_obs - y_pred)**2)
        r2 = round(1 - RSS/TSS,2)
        return r2

    def adj_r2_func(self, y_obs, y_pred):
        ''' Adjusted R2 metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        adj_r2: float
            adj_r2 metric
        '''
        if self.type == 'exp' and self.mult_v == False:
            adj_r2 = self.r2_func(y_obs, y_pred)
        else:
            n = y_obs.shape[0]
            p = self.X.shape[1]
            adj_r2 = round(1 - ((1 - self.r2_func(y_obs, y_pred)) * (n - 1))/(n-p-1), 2)
        return adj_r2
    
    def mae_func(self, y_obs, y_pred):
        ''' MAE metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        MAE: float
            MAE metric
        '''
        if self.type == 'exp':
            y_obs, y_pred = np.exp(y_obs), np.exp(y_pred)
        else:
            pass
        mae = round((np.sum(abs(y_obs - y_pred))/len(y_pred)), 2)
        return mae

    def mape_func(self, y_obs, y_pred):
        ''' MAPE metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        MAPE: float
            MAPE metric
        '''
        if self.type == 'exp':
            y_obs, y_pred = np.exp(y_obs), np.exp(y_pred)
        else:
            pass
        mape = round((np.sum(abs((y_obs - y_pred)/y_obs))/len(y_pred)), 2)
        return mape*100

    def rmse_func(self, y_obs, y_pred):
        ''' RMSE metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        RMSE: float
            RMSE metric
        '''
        if self.type == 'exp':
            y_obs, y_pred = np.exp(y_obs), np.exp(y_pred)
        else:
            pass
        rmse = round((sum((np.subtract(y_obs, y_pred))**2)/len(y_obs))**0.5,2)
        return rmse
    
    def nrmse_func(self, y_obs, y_pred):
        ''' NRMSE metric method
        Parameters
        ------------
        y_obs: array
            observed values (true values)
        y_pred: array
            estimated values (model output)
        Returns
        -----------
        NRMSE: float
            NRMSE metric
        '''
        if self.type == 'exp':
            y_obs, y_pred = np.exp(y_obs), np.exp(y_pred)
        else:
            pass
        rmse = (sum((np.subtract(y_obs, y_pred))**2)/len(y_obs))**0.5
        nrmse = round(rmse/mean(y_obs),2)
        return nrmse*100

    def scoring_func(self):
        ''' Scorer method uses sklearn make scorer method and allow
        to create the metric methods listed above.
        Parameters
        ------------
        self: instace
        Returns:
        ------------
        scoring: dict
            scoring dictionary with make_scorer as methods
        '''
        r2_score_t = make_scorer(self.r2_func, greater_is_better=True)
        adj_r2_score_t = make_scorer(self.adj_r2_func, greater_is_better=True)
        rmse_score = make_scorer(self.rmse_func, greater_is_better=False)
        nrmse_score = make_scorer(self.nrmse_func, greater_is_better=False)
        mae_score = make_scorer(self.mae_func, greater_is_better=False)
        mape_score = make_scorer(self.mape_func, greater_is_better=False)
        scoring = {'R²': r2_score_t, 
                   'Adjusted R²': adj_r2_score_t, 
                    'MAE': mae_score, 
                    'MAPE (%)': mape_score, 
                    'RMSE': rmse_score, 
                    'NRMSE (%)': nrmse_score}
        return scoring

    def boxplot_score(self, df, scores, palette):
        '''
        Boxplot plots method.
        Parameters
        -----------
        df: dataframe
            dataframe with metrics
        scores: list
            scores names
        palette: list
            colors to plot
        Returns
        ----------
        fig: image
            matplotlib plot
        '''
        cm = 1/2.54
        sns.set_context('paper')
        fig, axs = plt.subplots(ncols = 2, nrows=3, figsize=(30*cm, 30*cm))
        row, col = 0, 0
        for letter, score in zip(['a', 'b', 'c', 'd', 'e', 'f'], scores):
            e = sns.boxplot(ax = axs[row, col], data=df.melt(value_vars=["train_"+score, "test_"+score], value_name='score'), y='score', x='variable', palette=palette)
            e.set_xticklabels(['Train' + '\nn={:.0f}'.format(df["train_"+score].shape[0]) \
                               + '\nMedia: {:.2f}'.format(np.mean(df["train_"+score])) \
                               +'\nDesv. P.: {:.2f}'.format(np.std(df["train_"+score])),                                                     
                               'Test' + '\nn={:.0f}'.format(df["test_"+score].shape[0]) \
                               +'\nMedia: {:.2f}'.format(np.mean(df["test_"+score])) \
                               +'\nDesv. P.: {:.2f}'.format(np.std(df["test_"+score]))])
            if score.endswith("R²") or score.endswith("(%)"):
                e.set_ylabel(score)
            else:
                e.set_ylabel(score + ' ' + self.wqp_unit)
    
            e.set_xlabel('')
            e.spines['right'].set_visible(False)
            e.spines['left'].set_visible(True)
            e.spines['top'].set_visible(False)
            e.spines['bottom'].set_visible(True)
            e.yaxis.grid(color='gray', linestyle='dashed')
            e.yaxis.grid(color='gray', linestyle='dashed')
            e.set_title(letter)
            e.axhline(y = self.dummy_results.get(f"dummy_{score}"), c= 'red', linestyle='--')
            col +=1
    
            if col == 2:
                col = 0
                row+=1
                
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.4)
        
        return plt.show()
    
    
    def validation_plots(self, residuals):
        '''
        Plots to be presented in validation step.
        Parameters
        -----------
        residuals: np.array
            array with residuals
        Returns
        ----------
        fig: image
            matplotlib plot
        '''
    
        def res_hist_plot(self, residuals):
            array = residuals
            #fig, ax = plt.subplots(figsize=(10*cm ,10*cm))
            g = sns.histplot(ax = axs[0,0], x=array)
            g.set_xlabel('Resíduos')
            g.set_ylabel('Frequência')
            g.set_title('a')
            #plt.show()
            return g
            
        def res_lineplot(self, residuals):
            
            g=sns.scatterplot(x = residuals.index, y = residuals, color='b', marker='.', ax = axs[0,1])
            g.axhline(y=0, c= 'red', linestyle='--')
            g.spines['right'].set_visible(False)
            g.spines['left'].set_visible(True)
            g.spines['top'].set_visible(False)
            g.spines['bottom'].set_visible(True)
            g.yaxis.grid(color='gray', linestyle='dashed')
            g.set_ylabel('∆ '+ g.get_ylabel())
            g.set_xlabel('Índice')
            g.set_title('b')
            g.set_xlim(g.get_xticks()[0]-0.5, g.get_xticks()[-1]+0.5)
            g.set_ylim(g.get_yticks()[0], g.get_yticks()[0] *-1)
            return g  
            
        def res_qq_plot(self, residuals):
            array = residuals
            array_norm = (array - array.mean()) / array.std()
            array_norm=array_norm.sort_values(ascending=True)
            norm_dist = np.random.normal(loc=0, scale=1, size=len(array_norm))
            norm_dist.sort()
            ax = sns.scatterplot(x = norm_dist, y = array_norm, marker='.', color='b', ax = axs[1,0])
            ax.plot(norm_dist, norm_dist, c='red')
            lims = [norm_dist.min(), norm_dist.max()]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Quantis da Distribuição Normal")
            ax.set_ylabel("Quantis Observados")
            ax.set_aspect('equal')
            ax.set_title('c')
            return ax
            
        def pred_obs_scatter(self):
            h = sns.scatterplot(x = self.y_test, y=self.y_pred, color='darkgrey', ax = axs[1,1])      
            h.spines['right'].set_visible(True)
            h.spines['left'].set_visible(True)
            h.spines['top'].set_visible(True)
            h.spines['bottom'].set_visible(True)
            h.set_ylabel('Valores estimados de '+ h.get_xlabel())
            h.set_xlabel('Valores observados de '+ h.get_xlabel())
            h.set_title('Correlação entre valores estimados e observados')
            lims = [
                np.min([h.get_xlim(), h.get_ylim()]),  # min of both axes
                np.max([h.get_xlim(), h.get_ylim()]),  # max of both axes
            ]
            if lims[0]<0:
                lims[0]=0
            else:
                pass
            # now plot both limits against eachother
            h.plot(lims, lims, '--', alpha=0.5, zorder=0, color='gray')
            h.text(0.93, 0.9,'1:1', fontdict={'family': 'arial', 'color':  'black', 'weight': 'light', 'size': 10, 'rotation': '45'}, transform=h.transAxes)
            h.set_title('d')
            h.set_xticks(h.get_yticks())
            h.set_xlim(lims)
            h.set_ylim(lims)
            h.set_aspect('equal')
            h.text(0.02,
                   0.98,
                   "n = {} | R²: {:.2f} | AdjR²: {:.2f}".format(self.y_test.shape[0], self.val_r2, self.val_adj_r2),
                   fontdict={'ha': 'left', 'va': 'top', 'family': 'arial', 'color':  'black', 'weight': 'light', 'size': 10, 'rotation': '0'},
                   transform=h.transAxes)
            
            h.text(0.99,
                   0.01,
                   "MAE: " + "{:.1f} {} |".format(self.val_mae, self.wqp_unit) \
                   + " MAPE: " + "{:.0f} %".format(self.val_mape) \
                   + "\nRMSE: " + "{:.1f} {} |".format(self.val_rmse, self.wqp_unit) \
                   + " nRMSE: " + "{:.0f} %".format(self.val_nrmse),
                   fontdict={'ha': 'right', 'va': 'bottom', 'family': 'arial', 'color':  'black', 'weight': 'light', 'size': 10, 'rotation': '0'},
                   transform=h.transAxes)
            return h
        
        cm = 1/2.54
        fig, axs = plt.subplots(nrows= 2, ncols=2, figsize=(30*cm, 30*cm))
        res_hist_plot(self, residuals)
        res_lineplot(self, residuals)
        res_qq_plot(self, residuals)
        pred_obs_scatter(self)
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.2)
        return plt.show()
            
    def train_test_split_data(self, rs = None):
        self.test_rs = rs
        if self.type == 'linear':
            self.X, self.y = self.X, self.y
        elif self.type == 'poly':
            poly = PolynomialFeatures(degree = 2)
            self.X, self.y = poly.fit_transform(self.X), self.y
        elif self.type == 'exp':
            self.X, self.y = self.X, np.log(self.y)
        elif self.type == 'log':
            self.X, self.y = np.log(self.X), self.y
        elif self.type == 'standard_scaler':
            scaler = StandardScaler()
            self.X, self.y = scaler.fit_transform(self.X), self.y
        elif self.type == 'min_max':
            scaler = MinMaxScaler()
            self.X, self.y = scaler.fit_transform(self.X), self.y
        else:
            raise NameError('You probably forgot to enter a type. The types possible are: linear, poly, exp, poly, standard_scaler, min_max')
        
        if self.test_rs == None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)
            
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = self.test_rs)
        
    
    def grid_search(self, param_grid, scoring, refit, n_cv=5, r_t_c=True):
        '''
        Parameters
        ----------
        param_grid: dict
            Dictionary with model paramertes to be iterated
        scoring: dict
            All scoring methods to be used in model
        n_cv: int
            Number of Kfold in cross validation method
        refit: str
            Scoring method to return the best model
        r_t_c: bool
            Return train score
        '''
        if self.verbose == True:
            print("\n#--------------INICIANDO O GRID SEARCH CV------------#")
        else:
            pass
        
        self.scoring = scoring
        clf = GridSearchCV(self.model, 
                           param_grid = param_grid, 
                           scoring = self.scoring, 
                           cv = n_cv, 
                           refit = refit, 
                           return_train_score=r_t_c, 
                           verbose=0)
        clf.fit(self.X_train, self.y_train)
        temp_df=pd.DataFrame(clf.cv_results_, columns = sorted(clf.cv_results_.keys()))
        col = 'rank_test_' + refit
        self.params = temp_df.loc[temp_df[col].idxmin(), 'params']
        if self.verbose == True:
            print("\nO GridSearchCV encontrou os melhores hiperparâmetros \
                    para o modelo escolhido ({}), sendo eles: {}".format(self.model, self.params))
            print("\n#-----------FIM DO GRID SEARCH CV--------------#")
        else:
            pass
        del temp_df, clf
        
              
    def cv_model_results(self, n_cv=5, n_repeats=3, rs = None):
        self.cv_rs = rs
        if self.verbose == True:
            print("\n#--------------INICIANDO O CROSS VALIDATION PARA OS MELHORES HYPERPARAMETERS------------#")
        else:
            pass
        self.n_cv = n_cv
        self.n_repeats = n_repeats
        if self.cv_rs == None:
            cv = RepeatedKFold(n_splits=n_cv, n_repeats=n_repeats)
        else:
            cv = RepeatedKFold(n_splits=n_cv, n_repeats=n_repeats, random_state = self.cv_rs)
        self.len_train, self.len_test = [len(xs) for xs in list(cv.split(self.X_train))[0]]
        self.cv_results = cross_validate(self.model.set_params(**self.params), self.X_train, self.y_train, cv=cv,
                               scoring = self.scoring,
                               return_train_score = True,
                               verbose = 0,
                               return_estimator = True)
        dummy = DummyRegressor(strategy="mean").fit(self.X, self.y)
        dummy_pred = dummy.predict(self.X)
        self.dummy_results = {}
        self.dummy_results['dummy_R²'] = self.r2_func(self.y, dummy_pred)
        self.dummy_results['dummy_Adjusted R²'] = self.adj_r2_func(self.y, dummy_pred)
        self.dummy_results['dummy_MAE'] = self.mae_func(self.y, dummy_pred)
        self.dummy_results['dummy_MAPE (%)'] = self.mape_func(self.y, dummy_pred)
        self.dummy_results['dummy_RMSE'] = self.rmse_func(self.y, dummy_pred)
        self.dummy_results['dummy_NRMSE (%)'] = self.nrmse_func(self.y, dummy_pred)
        
        temp_df=pd.DataFrame(self.cv_results)
     
        ## arrumandos os valores negativos no dataframe
        for i in self.scoring.keys():
            if i == "R²" or i == "R2" or i == "r2" or i == "R_2" or i == "r_2" or i == "r2_score" or i =="Adjusted R²":
                pass
            else:
                temp_df.loc[:, "train_"+i] = temp_df.loc[:, "train_"+i] * -1
                temp_df.loc[:, "test_"+i] = temp_df.loc[:, "test_"+i] * -1
        
        
        temp_df = temp_df.sort_values(by = ['test_R²', 'train_R²'], ascending=False)
        if self.verbose == True:
            print("\n#-----------PLOTANDO GRÁFICOS COM MÉTRICAS DE TREINAMENTO E TESTE--------------#")
            print("\n#-----------FIM DO CROSS VALIDATION--------------#")
            self.boxplot_score(df = temp_df, scores = self.scoring.keys(), palette=['#A84269', '#4A97A8'])
        else:
            pass
        return temp_df
        
    
    def model_left_out(self):
        if self.verbose == True:
            print("\n#-----------ETAPA DE VALIDAÇÃO DO MODELO - APLICAÇÃO DO MODELO EM 20% DOS DADOS NÃO CALIBRADOS--------------#")
        else:
            pass
        best_model = self.model.set_params(**self.params)
        
        best_model.fit(self.X_train, self.y_train)
        
        self.y_pred = best_model.predict(self.X_test)
        
        if self.type == 'exp':
            residuals = np.exp(self.y_test) - np.exp(self.y_pred)
                
        else:
            residuals = self.y_test - self.y_pred

        self.val_r2 = self.r2_func(self.y_test, self.y_pred)
        self.val_adj_r2 = self.adj_r2_func(self.y_test, self.y_pred)
        self.val_mae = self.mae_func(self.y_test, self.y_pred)
        self.val_mape = self.mape_func(self.y_test, self.y_pred)
        self.val_rmse = self.rmse_func(self.y_test, self.y_pred)
        self.val_nrmse = self.nrmse_func(self.y_test, self.y_pred)
        
        if self.verbose == True:
            print("\n#-----------PLOTANDO GRÁFICOS DE VALIDAÇÃO--------------#")               
            self.validation_plots(residuals)
        else:
            pass

        self.result_dict['type'] = self.type
        self.result_dict['model'] = str(best_model).split("(")[0]
        if self.mult_v == True:
            self.result_dict['class'] = "Multivariate"
        else:
            self.result_dict['class'] = "Univariate"
        self.result_dict['attributes'] = str(self.X_name)
        self.result_dict['target'] = str(self.y_name)
        self.result_dict['params'] = str(self.params)
        self.result_dict['train_test_rs'] = self.test_rs
        self.result_dict['fit/test/validation']="{}/{}/{}".format(str(self.len_train), str(self.len_test), str(self.X_test.shape[0]))
        self.result_dict['KFoldRepeat']= "{} folds x {} repeats".format(self.n_cv, self.n_repeats)
        self.result_dict['cv_rs'] = self.cv_rs
        self.result_dict['metrics'] = ", ".join(list(self.scoring.keys())) 
        for idx, i in enumerate(self.scoring.keys()):
            if idx == 0:
                self.result_dict['mean_train_'+i] = round(np.nanmean(self.cv_results["train_"+i]),2)
                self.result_dict['mean_test_'+i] = round(np.nanmean(self.cv_results["test_"+i]),2)
                self.result_dict['validation_'+i] = self.val_r2
                self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_R²')
                
            elif idx == 1:
                self.result_dict['mean_train_'+i] = round(np.nanmean(self.cv_results["train_"+i]),2)
                self.result_dict['mean_test_'+i] = round(np.nanmean(self.cv_results["test_"+i]),2)
                self.result_dict['validation_'+i] = self.val_adj_r2
                self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_Adjusted R²')
                
            else:
                self.result_dict['mean_train_'+i] = round(np.nanmean(self.cv_results["train_"+i]*-1),2)
                self.result_dict['mean_test_'+i] = round(np.nanmean(self.cv_results["test_"+i]*-1),2)
                if i.startswith('MAE'):
                    self.result_dict['validation_'+i] = self.val_mae
                    self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_MAE')
                elif i.startswith('MAPE'):
                    self.result_dict['validation_'+i] = self.val_mape
                    self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_MAPE (%)')
                elif i.startswith('RMSE'):
                    self.result_dict['validation_'+i] = self.val_rmse
                    self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_RMSE')
                elif i.startswith('NRMSE'):
                    self.result_dict['validation_'+i] = self.val_nrmse
                    self.result_dict['baseline_'+i] = self.dummy_results.get('dummy_NRMSE (%)')
        return self.result_dict

"""
Reading DF
"""
df = pd.read_csv(r'dados_pampulha_concatenados_mar_maio_jun_jul_ago_set.csv')
df['DateTime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df[df['Distance'] < 10]
df[df['Gimbal Degree'].between(80, 100)]
df[pd.isna(df['Gimbal Degree'])]['date'].value_counts()

"""
Removing outlier
"""
def filter_bands(x):
  s = []
  for i in x:
    if i < 0.2: #change if necessary
      s.append(i)
    else:
      s.append(np.nan)
  return s

df.loc[:, ['Blue', 'Green', 'Red', 'Near-IR', 'RedEdge']]=df.loc[:, ['Blue', 'Green', 'Red', 'Near-IR', 'RedEdge']].apply(filter_bands)

df.dropna(subset=['Blue', 'Green', 'Red', 'Near-IR', 'RedEdge'], inplace=True, axis=0)



"""
Columns to work with
"""
rrs_columns = ['Blue', 'Green', 'Red', 'RedEdge', 'Near-IR']


index_columns = ['KIVU', 'Kab 1(Rs)', 'SABI', 'NDCI', 'NDVI', 
                     '2BDA_1', '2BDA_2', '3BDA_1', '3BDA_MOD',
                     'B3B1', 'GB1', 'GR']

rrs_index_columns = ['Blue', 'Green', 'Red', 'RedEdge', 'Near-IR', 
                     'KIVU', 'Kab 1(Rs)', 'SABI', 'NDCI', 'NDVI', 
                     '2BDA_1', '2BDA_2', '3BDA_1', '3BDA_MOD',
                     'B3B1', 'GB1', 'GR']

wqp_columns = ['secchi_cm', 'temp_c', 'chl_rfu', 'pc_rfu',
               'fdom_rfu', 'turb_fnu', 'sdt_mgl']

main_columns = ['Image', 'spot', 'DateTime', 'time', 'Distance']

"""
Pearson correlation
"""

#band_manipulation_cols = list(df_selected.iloc[:, 56:].columns)


potential_independent_variables = rrs_index_columns

cm=1/2.54
sns.set_context("paper")
corr = df.corr(method='pearson')

corr = corr.loc[potential_independent_variables, wqp_columns]
corr = corr[abs(corr)>0.4].dropna(how='all', axis=0)


#fig, ax = plt.subplots(figsize=(20*cm,20*cm))

#ax = sns.heatmap(corr, annot=True, cmap='coolwarm')

"""
ML Models
"""

df_selected = df.copy(deep=True)

cm=1/2.54
sns.set_context("paper")
corr = df_selected.corr(method='pearson')

fig, ax = plt.subplots(figsize=(30*cm,10*cm))

ax = sns.heatmap(corr.loc[wqp_columns, rrs_columns + index_columns], annot=True, cmap='coolwarm')

"""
Attribute Selection
"""

attributes_wqp = ('secchi_cm', 'chl_rfu', 'pc_rfu',
               'fdom_rfu', 'turb_fnu')
units_wqp = ('cm', 'rfu', 'rfu', 'rfu', 'fnu')

df_folder = ('secchi', 'chl', 'pc', 'fdom', 'turb')
csv_filenames = ('secchi', 'chl', 'pc', 'fdom', 'turb')

for k in range(len(attributes_wqp)):
    indep_columns = rrs_columns + index_columns
    wqp = attributes_wqp[k] #select which attribute will be used
    wqp_unit = units_wqp[k] #select measuring parameter
    corr_secchi = corr.loc[indep_columns, [wqp]]
    corr_secchi = corr_secchi[abs(corr_secchi)>=0.6].dropna(how='all', axis=0)
    # ax = sns.heatmap(corr_secchi.T, annot=True, cmap='coolwarm')

    """
    Linear Regression - Univariate
    """

    list_models = [LinearRegression()]
    list_param_grid = [{'positive': [False, True]}]

    independent_variables = rrs_columns + list(corr_secchi.index)

    df_secchi = pd.DataFrame()

    i=0

    for m_type in ['linear', 'poly', 'exp', 'log']:
        for attr in independent_variables:
            X = df_selected[df_selected[wqp].notna()][[attr]]
            y = df_selected[df_selected[wqp].notna()][wqp]
            for model, param_grid in zip(list_models, list_param_grid):
                try:
                    start_time = time.time()
                    print("#-----------------------TYPE: {}------------------#".format(str(m_type)))
                    print("#-----------------------ATTRIBUTE: {}------------------#".format(str(attr)))
                    print("#-----------------------MODEL: {}------------------#".format(str(model)))
                    model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = False, m_type = m_type, verbose=False)
                    model_.train_test_split_data(rs = 4)
                    scoring = model_.scoring_func()
                    model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                    model_.cv_model_results(n_cv=5, n_repeats=100)
                    result=model_.model_left_out()
                    result = pd.DataFrame(pd.Series(result, name=i)).T
                    df_secchi = pd.concat([df_secchi, result], axis = 0)
                    print("#-----------------------TIME {:.2f}s------------------#".format(time.time() - start_time))
                    print("#-----------------------END OF {}------------------#".format(str(model)))
                except:
                    print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, attr, m_type, model))
                i+=1

    """
    Linear Regression - Multivariate
    """

    list_models = [LinearRegression()]
    list_param_grid = [{'positive': [False, True]}]

    independent_variables = list(corr_secchi.index)
    X = df_selected[df_selected[wqp].notna()][independent_variables]

    i=0
    for m_type in ['linear', 'poly', 'exp', 'log', 'standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs = 4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, rrs_index_columns, m_type, model))
            i+=1  

    """
    Save result 
    """
    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_lr' + '.csv')

    """
    Lasso - Univariate
    """

    list_models = [Lasso()]
    list_param_grid = [{'alpha': [0.1, 0.5, 1, 10, 100]}]

    independent_variables = list(corr_secchi.index)

    df_secchi = pd.DataFrame()

    i=0

    for m_type in ['linear', 'poly', 'exp', 'log']:
        for attr in independent_variables:
            X = df_selected[df_selected[wqp].notna()][[attr]]
            y = df_selected[df_selected[wqp].notna()][wqp]
            for model, param_grid in zip(list_models, list_param_grid):
                try:
                    start_time = time.time()
                    print("#-----------------------TYPE: {}------------------#".format(str(m_type)))
                    print("#-----------------------ATTRIBUTE: {}------------------#".format(str(attr)))
                    print("#-----------------------MODEL: {}------------------#".format(str(model)))
                    model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = False, m_type = m_type, verbose=False)
                    model_.train_test_split_data(rs = 4)
                    scoring = model_.scoring_func()
                    model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                    model_.cv_model_results(n_cv=5, n_repeats=100)
                    result=model_.model_left_out()
                    result = pd.DataFrame(pd.Series(result, name=i)).T
                    df_secchi = pd.concat([df_secchi, result], axis = 0)
                    print("#-----------------------TIME {:.2f}s------------------#".format(time.time() - start_time))
                    print("#-----------------------END OF {}------------------#".format(str(model)))
                except:
                    print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, attr, m_type, model))
                i+=1

    """
    Lasso - Multivariate
    """

    list_models = [Lasso()]
    list_param_grid = [{'alpha': [0.1, 0.5, 1, 10, 100]}]

    independent_variables = list(corr_secchi.index)
    X = df_selected[df_selected[wqp].notna()][independent_variables]

    i=0
    for m_type in ['linear', 'poly', 'exp', 'log', 'standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs=4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, independent_variables, m_type, model))
            i+=1  
        
    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_lasso' + '.csv')

    """
    SVR - Multivariate
    """

    list_models = [SVR()]
    list_param_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [500, 1000, 10000], 'epsilon': [1, 2, 5, 10], 'gamma': ['scale', 'auto']}]

    df_secchi = pd.DataFrame()
    independent_variables = list(corr_secchi.index)
    X = df_selected[df_selected[wqp].notna()][rrs_columns]

    y = df_selected[df_selected[wqp].notna()][wqp]


    i=0
    for m_type in ['linear', 'poly', 'exp', 'log', 'standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs = 4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, independent_variables, m_type, model))
            i+=1  

    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_svr' + '.csv')

    """
    Random Forest - Multivariate
    """

    list_models = [RandomForestRegressor()]
    list_param_grid = [{'n_estimators': [10,50,100,500,1000], 'criterion': ["squared_error"], 'max_depth': [2, 10, 25, 50]}]

    df_secchi = pd.DataFrame()

    independent_variables = list(corr_secchi.index)

    X = df_selected[df_selected[wqp].notna()][independent_variables]

    y = df_selected[df_selected[wqp].notna()][wqp]


    i=0
    for m_type in ['linear', 'poly', 'exp', 'log', 'standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs=4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, independent_variables, m_type, model))
            i+=1  

    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_rf' + '.csv')

    """
    XGBOOST
    """

    list_models = [XGBRegressor()]
    list_param_grid= [{'n_estimators': [10,50,100,500,1000], 
                'max_depth': [2, 10, 25, 50]}]
    df_secchi = pd.DataFrame()

    independent_variables = list(corr_secchi.index)

    X = df_selected[df_selected[wqp].notna()][independent_variables]

    y = df_selected[df_selected[wqp].notna()][wqp]


    i=0
    for m_type in ['linear', 'poly', 'exp', 'log', 'standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs=4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, independent_variables, m_type, model))
            i+=1
    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_xgb' + '.csv')

    """
    MLP Regressor
    """

    list_models = [MLPRegressor()]
    list_param_grid=[{'hidden_layer_sizes': [(3, ), (3, 3)],
                'activation': ['tanh', 'relu'], 
                'solver': ['sgd'], 
                'max_iter': [100000], 
                'learning_rate': ['adaptive', 'invscaling']}]


    df_secchi = pd.DataFrame()

    independent_variables = list(corr_secchi.index)

    X = df_selected[df_selected[wqp].notna()][independent_variables]

    y = df_selected[df_selected[wqp].notna()][wqp]


    i=0
    for m_type in ['standard_scaler', 'min_max']:       
        for model, param_grid in zip(list_models, list_param_grid):    
            try:             
                print("#-----------------------MODEL: {}------------------#".format(str(model)))
                model_ = Model(X, y, model = model, wqp_unit=wqp_unit, mult_v = True, m_type = m_type, verbose = False)
                model_.train_test_split_data(rs=4)
                scoring = model_.scoring_func()
                model_.grid_search(param_grid = param_grid, scoring = scoring, n_cv=5, refit = 'RMSE')
                t = model_.cv_model_results(n_cv=5, n_repeats=100)
                result = model_.model_left_out()
                result = pd.DataFrame(pd.Series(result, name=i)).T
                df_secchi = pd.concat([df_secchi, result], axis = 0)
                print("#-----------------------END OF {}------------------#".format(str(model)))
            except:
                print("Couldn't run for the specific variables: \n{}\n{}\n{}\n{}".format(wqp, independent_variables, m_type, model))
            i+=1

    df_secchi.to_csv('./resultados_modelagem/'+ df_folder[k] +'/' + csv_filenames[k] + '_mlp' + '.csv')

"""
Análise de Resultados
"""
dis_list = ('resultados_modelagem/secchi', 'resultados_modelagem/chl', 'resultados_modelagem/pc', 'resultados_modelagem/fdom', 'resultados_modelagem/turb', )

results = pd.DataFrame()

for z in range(len(dis_list)):
    for file in os.listdir(dis_list[z]):
        if file == '.ipynb_checkpoints':
            pass
        else:
            file_2 = os.path.join(dis_list[z], file)
            df_s = pd.read_csv(file_2)
            results = pd.concat([results, df_s], axis = 0)
    results.reset_index(drop=True, inplace=True)

    results.loc[results['mean_test_RMSE'].idxmin(), :]
    results.to_csv('./resultados_modelagem/' + df_folder[z]+'/' + csv_filenames[z] + '_meanRMSE' + '.csv')
