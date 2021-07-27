import numpy as np
import pandas as pd
from math import *
import sys
import matplotlib.pyplot as plt
from _products.visualization_tools import *
from __Keras_Tools_.keras_tools import add_keras_layers
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from _products._DEEPSOLAR_ import Data_set_paths, G_Z_scaler, Xu_Scalable, Block_Groups
from _products.utility_fnc import *
from sklearn import metrics
from keras.layers import Dense, Dropout, Input, BatchNormalization, concatenate, Concatenate
from keras.models import Sequential, Model
from keras import optimizers
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
viz = Visualizer()

# =========================================================================
# =========================================================================
#                  TODO:Feature Selection  and Preprocessing tools
# =========================================================================
# =========================================================================

def LCN_transform(df_base, target='Adoption', mtc=0, mxcc=1, corrs=('kendall', 'pearson'), inplace=False, verbose=False):
    """ This will perform a LCN search and either return a reduced data frame
        or reduce the given
    :param df: data frame
    :param target: the target('s) of the analysis
    :param mtc:    minimum target correlation
    :param mxcc:   maximum cross correlation between independent variables
    :param corrs:  types of correlation matrices to create
    :param inplace: if true will modify original, otherwise returns a reduced version
    :return:
    """
    try:
        corr_dfk = df_base.corr(method=corrs[0])
    except:
        print('There is some issue trying to get the correlation matrix for {} style correlation for target...'.format(corrs[0]))
        rl = df_base.columns.tolist()
        del rl[rl.index(target)]
        return df_base[rl], df_base.loc[:, target]
    try:
        corr_df = df_base.corr(method=corrs[1])
    except:
        print('There is some issue trying to get the correlation matrix for {} style correlation for features...'.format(corrs[1]))
        rl = df_base.columns.tolist()
        del rl[rl.index(target)]
        return df_base[rl], df_base.filter(items=[target])
    # if didn't hit any exceptions move on
    corr_df[target] = corr_dfk[target].values.flatten()
    corr_df.loc[target, :] = corr_dfk.loc[target,:]

    print('correlation matrix:')
    print(pd.DataFrame(corr_df.loc[target, :], columns=[target], ).sort_values(target))
    print('---------------')
    print()

    # TODO: this was added 7/31/20
    corr_df = np.abs(corr_df)

    dfattribs = list(df_base.columns.values.tolist()).copy()
    del dfattribs[dfattribs.index(target)]
    print('predictor variables:')
    print(dfattribs)
    print()
    lcn_d = LCN(corr_df, target=target)

    rl = HCTLCCL(lcn_d, [], target=target, options=dfattribs, target_corr_lim=mtc,
                 cross_corr_lim=mxcc)

    if verbose:
        print()
        print('return features:')
        print(rl)
        print()

    #return df_base.loc[:, rl], df_base.loc[:, [target]]
    return df_base.loc[:, rl], df_base.filter(items=[target])

# sorts a correlation and
def LCN(corr_M, threshold=100, target='Adoption'):
    """Takes a correlation matrix some threshold value and the name of the target column.
        This method will use the given correlation matrix to create a dictionary for all features
        in the matrix where the keys are the features and the values are a dictionary of all other
         features as keys and vals are the key features correlations to those variables....
            dict = {'feat1': {'feat2: correl_feat1_feat2}}
        result is a dictionary keyed on the features, with values of a sorted dictionary keyed on other
         features sorted on correlation
    :param corr_M: correlation matrix
    :param threshold: TODO: I don't remember exactly what this does
    :param target:
    :return:
    """
    # go through Data frame of correlations grabbing the list and sorting from lowest to
    # grab the attribs
    attribs = corr_M.columns.values.tolist()
    lv1_d = {}
    for ata in attribs:
        lv1_d[ata] = dict()
        for atb in attribs:
            if ata != atb and atb != target:
                if corr_M.loc[ata, atb] < threshold:
                    lv1_d[ata][atb] = abs(corr_M.loc[ata, atb])
        if ata == target:
            lv1_d[ata] = sort_dict(lv1_d[ata], reverse=True)
        else:
            lv1_d[ata] = sort_dict(lv1_d[ata], reverse=False)

    return lv1_d
def HCTLCCL(corr_dic, start_vars, target, options, target_corr_lim = .09, cross_corr_lim=.55):
    rl = list(start_vars)
    for p_var in  corr_dic[target]:
        if abs(corr_dic[target][p_var]) > target_corr_lim and p_var not in rl and p_var in options:
            rl = check_correlation(p_var, corr_dic, rl, '', cross_corr_lim)
        if abs(corr_dic[target][p_var]) < target_corr_lim:
            return rl
    return rl
def check_correlation(check_var, corr_dic, cl, used, cross_corr_lim = .55):
    # go through current list
    for variable in cl:
        # checking cross correlation between the possible
        # variable to be added and the current one from
        # the current list if it surpasses the threshold
        # return the current list as is
        if abs(corr_dic[check_var][variable]) > cross_corr_lim:
            return cl
    # if correlations with all current variables
    # are within limits return the current list
    # updated with the new value
    return cl + [check_var]
def forward_selector_test(x, y, x2, y2):
    pass
# =========================================================================
# =========================================================================
#                  TODO: Statistics and Preprocessing
# =========================================================================
# =========================================================================
class NORML():
    """a data normalizer. has two normalization methods
        1) min max normalization:
            * equation : (X-Min_val)/(Max_val - Min_val)
            * rescales the data to [0,1] values
            * centers pdf on the mean
        2) z standardization (X-Min_val)/(Max_val - Min_val)
            * equation : (X-Mean)/(Standard_Deviation)
            * rescales the data to [-1,1] values
            * centers pdf on 0 with std = 1
        Select the type by setting the input argument
        nrmlz_type to:
            * : minmax for option 1
            * : zstd for option 2
    """
    def __init__(self, nrmlz_type='minmax'):
        self.mu=None
        self.std=None
        self.cov=None
        self.corr=None
        self.cov_inv=None
        self.cov_det=None
        self.min = None
        self.max = None
        self.normlz_type=nrmlz_type

    def set_type(self, n_type):
        self.normlz_type = n_type

    def fit(self, df=None, X=None):
        if type(df) != type(np.array([0])):
            df = pd.DataFrame(df)
        self.process_df(df)

    def process_df(self, df):
        self.mu = df.values.mean(axis=0)
        self.std = df.values.std(axis=0)
        self.min = df.min()
        self.max = df.max()
        self.cov = df.cov()
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def transform(self,df, headers=None):
        if type(df) !=type(np.array([0])):
            df = pd.DataFrame(df)
        if self.normlz_type == 'zstd':
            if headers is not None:
                return pd.DataFrame((df - self.mu) / self.std, columns=headers)
            return pd.DataFrame((df - self.mu) / self.std)
        elif self.normlz_type == 'minmax':
            if headers is not None:
                return pd.DataFrame((df - self.min) / (self.max - self.min), columns=headers)
            return pd.DataFrame((df - self.min) / (self.max - self.min))

def standardize_data(X, X2, scaler_ty='minmax'):
    #scaler_ty = 'std'
    if scaler_ty == 'minmax':
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(X)
        Xtrn = mm_scaler.transform(X)
        Xtsn = mm_scaler.transform(X2)
    elif scaler_ty == 'std':
        std_scaler = StandardScaler()
        std_scaler.fit(X)
        Xtrn = std_scaler.transform(X)
        Xtsn = std_scaler.transform(X2)
    return Xtrn, Xtsn

#def cross_val_splitter(X, y, tr=.5, ts=.5, vl=0, seed=None, verbose=False, target=None):
#    train_idx, val_idx, test_idx = split_data(X, y, p_train=tr, p_test=vl, p_val=ts, verbose=verbose,seed=seed)


def cross_val_splitterG(df0, rl, target='Adoption', ts=.5, verbose=False, stratify=True):
    from sklearn.model_selection import train_test_split
    targets0 = df0[target]
    print('targets:')
    print(targets0)
    print('the features: {}'.format(rl))
    print('------------------------------')
    print('------------------------------')
    #quit(197)
    #df0 = df0.loc[:, rl]
    df0 = df_select(df0, rl)
    tr = 1 - ts
    # Create training and testing sets for the data
    if stratify:
        X_train0, X_test0, y_train0, y_test0 = train_test_split(df0, targets0, stratify=targets0, test_size=ts,
                                                                train_size=tr)
    else:
        X_train0, X_test0, y_train0, y_test0 = train_test_split(df0, targets0, test_size=ts,
                                                                train_size=tr)
    if verbose:
        print('Training:')
        print(X_train0.describe())
        print('Testing:')
        print(X_test0.describe())
    print('------------------')
    print('------------------')
    print(y_train0)
    print(y_test0)
    print('------------------')
    print('------------------')
    return (X_train0, y_train0), (X_test0, y_test0)

def quick_rf_scorer(ytruth, ypred, feat_imp, feats, target='Adoption', cmap='summer',
                    cm_title='Confusion Matrix:\nacc: {}\nsen: {}\nspe: {}',
                    fi_title='Feature importance for {}',
                    figsizeCM=(10, 10), figsizeFI=(20,20), ):
        feat_dic = display_significance(feat_imp, feats)
        cm = confusion_matrix(ytruth, ypred)
        negc = cm[0].sum()
        posc = cm[1].sum()
        sen = cm[1][1] / posc
        spe = cm[0][0] / negc
        rd = {}
        rd['acc'] = np.around(accuracy_score(ytruth, ypred)*100, 2)
        rd['sen'] = np.around( sen * 100, 2)
        rd['spe'] = np.around( spe * 100, 2)
        plt.figure(figsize=figsizeCM)
        plt.title(cm_title.format(rd['acc'], sen, spe))
        plt.imshow(cm, cmap=cmap)
        plt.colorbar()

        feat_df = pd.DataFrame(feat_dic, index=[0]).filter(items=list(feat_dic.keys())[:20])

        width=.4
        x_cntrs = np.arange(len(list(feat_dic.keys())[:20]))
        feat_vars, feat_vals = list(feat_dic.keys())[:20], list(feat_dic.values())[:20]
        fig, ax = plt.subplots(1, 1, figsize=figsizeFI)
        cnt = 0
        for f, v, x in zip(feat_vars, feat_vals, x_cntrs):
            print('{}, {}, {}'.format(f, v, x))
            ax.barh(x, width=v, height=width,  label=f)
            """
            if cnt == 0:
                #ax.barh(x, v, width=width, label=f)
                ax.barh(x, v, label=f)
                cnt += 1
            else:
                ax.barh(x, v, label=f)
            """
        ax.set_yticklabels(list(feat_dic.keys())[:20], rotation=0, )
        ax.set_yticks(x_cntrs)
        ax.set_ylim(0 - (width / 2) * 4, len(x_cntrs))
        ax.set_xlabel('Feature Importance Value')
        ax.legend()



def split_data(X, y, p_train=.70, p_test=.30, p_val=.0, priors = None, verbose=False, seed=False, lr=True):
    """Returns a randomized set of indices into an array for the purposes of splitting data"""
    dXY = None
    if type(X) != type(pd.DataFrame([0])):
        nx = pd.DataFrame(X)
        nx[X.shape[1]] = y.values.tolist()
        dXY = nx.values
        np.random.shuffle(dXY)
    N = len(X)

    train = int(np.around(N * p_train, 0))
    if p_val == 0:
        test = int(np.around(N * p_test, 0, ))
        val = 0
    else:
        test = N - train
        val = N - train - test

    tr = dXY[0:train]
    ts = dXY[train:train+test]
    if p_val != 0:
        vl = dXY[train+test:]

    tr_X, tr_y = tr[:][0:len(dXY[0])], tr[:][len(dXY[0])]
    ts_X, ts_y = ts[:][0:len(dXY[0])], ts[:][len(dXY[0])]
    vl_X, vl_y = list(), list()
    if p_val != 0:
        vl_X, vl_y = tr[:][0:len(dXY[0])], tr[:][len(dXY[0])]
    '''
    if priors is not None:
        print('need to set up the distribution of the weights')

    if verbose:
        print('train set size: ', train)
        print('test set size: ', test)
        print('val set size: ', val)
    np.random.shuffle(X)
    tr_idx = rc

    for i in range(0, train):
        trn_idx.append(r_c[i])

    for i in range(train, train+test):
        tst_idx.append(r_c[i])

    for i in range(train+test, data_size):
        val_idx.append(r_c[i])

    if val == 0:
        return trn_idx, tst_idx
    else:
        return trn_idx, tst_idx, val_idx
    '''
    if p_val != 0:
        return (tr_X, tr_y), (ts_X, ts_y), (vl_X, vl_y)
    return (tr_X, tr_y), (ts_X, ts_y), (vl_X, vl_y)

def gstandardize_data(X, X2, scaler_ty='minmax'):
    if scaler_ty == 'minmax':
        nrml = NORML()
        nrml.fit(X)
        Xr = nrml.transform(X)
        xrts = nrml.transform(X2)
    if scaler_ty == 'zstd':
        nrml = NORML(scaler_ty=scaler_ty)
        nrml.fit(X)
        Xr = nrml.transform(X)
        xrts = nrml.transform(X2)
    return Xr, xrts

def normalize(X, mu, std, min, max, type='z', copy=False):
    if type == 'z':
        return z_normalize(X, mu, std)
    elif type == 'mm':
        return min_max_normalize(X, min, max)

def z_normalize(X, mu, std):
    return pd.DataFrame((X - mu)/std, columns=X.columns)

def min_max_normalize(X, min, max):
    return pd.DataFrame((X - min)/(max - min), columns=X.columns)


# =========================================================================
# =========================================================================
#                            TODO: Modeling tools
# =========================================================================
# =========================================================================
#Classification Model
class CMODEL():
    """a representation of a model for machine learning can in take in multiple data sets and perform
       a column wise merge
    """
    def __init__(self, file_list, exclude_list, target, df=None, usecols=None, usecol_list=None, verbose=False, lcn=False,
                 labeled=True, joins=('fips', 'fips', 'fips'), impute='drop', nas=(-999, ), drop_joins=False,st_vars=[],
                 mtc=.0, mxcc=1, dim_red=None, split_type='tt', tr_ts_vl = (.6, .4, 0), normal =None, complx=False):
        self.normlz = normal
        self.target = target                # the current models classification objective
        self.classes = list()               # the class values for this model
        self.model_mean = None              # the attribute mean values
        self.model_std = None               # the attribute std values
        self.model_cov = None
        self.model_cov_det = None
        self.model_cov_inv = None
        self.class_splits = dict()          # the data set split by class
        self.class_counts = dict()          # a count for each class in the data set
        self.class_priors = dict()          # the prior probaility of each class initialized by data set
        self.class_means = dict()           # the attribute means for each class
        self.class_std = dict()             # the attribute std for each class
        self.class_cov = dict()             # the covariance matrix for each class
        self.class_cor = dict()
        self.class_cov_inv = dict()         # the covariance matrix inverse for each class
        self.class_cov_det = dict()         # the class covariance matrix determinant
        self.attribs = None                 # the names of the attributes by column
        self.excluded=None                  # the excluded variables, can be added back as onehot encoded versions
        self.data_set=None                  # holds the desired data set
        self.og_dataset= None               # the merged set before any preprossing
        self.data_corr = None
        self.X = None                       # data or independent variables
        self.y = None                       # the target values or dependent variable
        self.Xtr_n=None
        self.Xts_n=None
        self.corr = None                    # the correlation matrix for the data
        self.Xfld = None                    # an fld transformed version of the data
        self.Xpca = None                    # a pca transformed version of the data
        self.dim_red = DimensionReducer()   # the models dimension reducer
        self.Xts=None
        self.yts=None
        self.complx=complx
        self.process_files(file_list, exclude_list, target, usecols, usecol_list, verbose, labeled, joins,
                           impute, nas, drop_joins=drop_joins, lcn=lcn, mtc=mtc, mxcc=mxcc, tr_ts_vl=tr_ts_vl,
                           df=df, st_vars=st_vars)

    def process_files(self, file_list, exclude_list, target, usecols, usecol_list, verbose, labeled,
                      joins=('fips', 'fips', 'fips'), impute='drop', nas=(-999,), drop_joins=False,
                      lcn=False,  mtc=.1, mxcc=.4, tr_ts_vl=(.6, .4, 0), df=None, to_encode=None, drops=None, st_vars=[]):
        df_list = list([])
        # go through and create and clean up data frames
        # dropping those in the exclude list
        doit=True
        if drops is None:
            tormv = list()
        else:
            tormv = drops
        #  If there was a data frame passed
        if df is not None:
            self.og_dataset = df
            self.excluded = tormv
            if to_encode is not None:
                hold_over = df.low[:, to_encode]
        else:
            # loop to set up input do data merger
            for df, ex in zip(file_list, exclude_list):
                print('Data file:', df)
                print('adding to be excluding', ex)
                df_list.append(pd.read_excel(df))
                tormv += ex
                #if doit and len(ex) > 0:
            self.excluded = tormv
            self.og_dataset = data_merger(df_list, joins=joins, verbose=verbose, drop_joins=True, target=target)

        #print(self.og_dataset)
        merged = self.og_dataset.drop(columns=tormv, inplace=False)
        self.data_corr = merged.sort_values(by=target, axis='index', ascending=False).corr(method='kendall')
        if usecols is not None:
            merged = merged.loc[:, usecols]
            self.data_corr = merged.corr(method='kendall')
        if lcn:
            self.data_corr = merged.corr(method='kendall')
            lcn_d = LCN(self.data_corr, target=target)
            rl = HCTLCCL(lcn_d, st_vars, target=target, options=merged.columns.values.tolist(), target_corr_lim=mtc,
                         cross_corr_lim=mxcc)
            merged = merged.loc[:, rl + [target]]
            print('columns used:')
            print(merged.columns)
        if impute == 'drop':
            for n in nas:
                merged.replace(n, np.nan)  # this value is used by the SVI data set to represent missing data
            merged = merged.dropna()
        self.data_set = merged
        self.attribs = merged.columns.values.tolist()
        print(self.attribs)
        del self.attribs[self.attribs.index(self.target)]
        self.X = pd.DataFrame(merged.loc[:, self.attribs], columns=self.attribs)
        self.y = pd.DataFrame(merged.loc[:, self.target], columns=[self.target])
        # TODO: NOW SPLIT THE DATA INTO DESIRED NUMBER OF FOLDS
        targets0 = self.y[target]
        ts = tr_ts_vl[1] + tr_ts_vl[2]
        print('ts size', ts)
        tr = 1 - ts
        # Create training and testing sets for the data
        X_train0, X_test0, y_train0, y_test0 = train_test_split(self.X, targets0, stratify=targets0, test_size=ts,
                                                                train_size=tr, )
        self.train_counts = y_train0.value_counts(normalize=True)
        self.test_counts = y_test0.value_counts(normalize=True)
        self.X, self.y = pd.DataFrame(X_train0, columns=self.attribs), pd.DataFrame(y_train0, columns=[self.target])
        self.Xts, self.yts = pd.DataFrame(X_test0, columns=self.attribs), pd.DataFrame(y_test0, columns=[self.target])
        self.N = self.X.shape[0]
        self.Nts = self.Xts.shape[0]
        self.d = self.X.shape[1]
        self.corr = self.X.corr()

        if self.normlz is not None:
            print('Normalize', self.normlz)
            self.X, self.Xts = standardize_data(self.X, self.Xts, scaler_ty=self.normlz)
            self.X = pd.DataFrame(self.X, columns=self.attribs)
            self.Xts = pd.DataFrame(self.Xts, columns=self.attribs)
            self.y.index = self.X.index
            self.yts.index = self.Xts.index
            print('y len', len(self.y.values))
            print('X len', len(self.X.values))
        self.grab_model_stats()
        # TODO: need to a some time move LCN stuff here
        # grab class specific stats
        self.calculate_class_stats()
        self.model_data = list((self.X, self.y))                    # store data and labels together in lit



    def grab_model_stats(self):
        print(self.X.values)
        print(self.X)
        self.model_mean = self.X.values.mean(axis=0)
        self.model_std = self.X.values.std(axis=0).mean()
        self.model_cov  = self.X.cov()
        if self.complx:
            self.model_cov_inv = np.linalg.inv(self.model_cov)
            self.model_cov_det = np.linalg.det(self.model_cov)
    def calculate_class_stats(self):
        self.classes = list(set(self.y[self.target]))
        print('classes', self.classes)
        for c in self.classes:
            self.splits_priors(c)
            self.class_means_std(c)
            if self.complx:
                self.class_cov_inv_det(c)
                self.class_cor[c] = self.class_splits[c].corr()
    def splits_priors(self,c):
        self.class_splits[c] = self.X.loc[self.y[self.target] == c, :]
        self.class_counts[c] = self.class_splits[c].shape[0]
        self.class_priors[c] = self.class_splits[c].shape[0] / self.N
    def class_means_std(self,c):
        self.class_means[c] = self.class_splits[c].values.mean(axis=0)
        self.class_std[c] = self.class_splits[c].values.std(axis=0)
    def class_cov_inv_det(self, c):
        self.class_cov[c] = self.class_splits[c].cov()
        if self.complx:
            self.class_cov_inv[c] = np.linalg.inv(self.class_cov[c].values)
            self.class_cov_det[c] = np.linalg.det(self.class_cov[c].values)
    def show_data_report(self):
        print('=======================================================================================')
        print('Train data size:', self.X.shape[0])
        print('y_train class distribution 0')
        print(self.train_counts)
        print('Test data size:', self.Xts.shape[0])
        print('y_test class distribution 0')
        print(self.test_counts)
        print('=======================================================================================')
        print('Features in Model:')
        print(self.X.columns.values)
        print('Predicting for {}'.format(self.target))
    def Reduce_Dimension(self, dr_type='FLD', pov=None, pc=None):
        if dr_type == 'FLD':
            self.perform_FLD()
    def perform_FLD(self):
        self.dim_red.fld_fit(self)
        self.Xfld = self.dim_red.FLD(self.X)
        self.tsXfld = self.dim_red.FLD(self.X)

class RMODEL():
    def __init__(self, X=None, Y=None, dataframe=None, columns=None, impute=None, verbose=False, trtsspl=(.5, ),
                 n_type=None, classes=(0,1), target='target', justify=True):
        #self.X = np.array(X)
        #self.Y = np.array(Y)
        if dataframe is None:
            if isinstance(X, type(pd.DataFrame)):
                columns = columns
                self.X = X
            else:
                self.X = pd.DataFrame(np.array(X))
            if isinstance(Y, type(pd.DataFrame())):
                target = Y.columns.tolist()[0]
                self.Y = Y
                Y = Y.values.tolist()
            else:
                self.Y = Y
            if columns is None:
                self.dataframe = pd.DataFrame(self.X)
            else:
                self.dataframe = pd.DataFrame(self.X, columns=columns)
            self.dataframe[target] = Y
            self.str_col = self.find_string_cols(self.dataframe)
        else:
            dataframe = dataframe.copy()
            np.random.shuffle(dataframe.values)
            self.X = pd.DataFrame(dataframe[columns])
            self.Y = pd.DataFrame(dataframe[target])
            self.dataframe=pd.DataFrame(dataframe[columns + [target]])

            str_col = self.find_string_cols(self.dataframe)
        self.str_col = str_col
        self.Xtr = None
        self.classes = classes
        self.ytr = None
        self.Xts = None
        self.yts = None
        self.train_test_split = trtsspl
        self.target = target
        self.columns = columns
        self.impute=impute
        self.justify=justify
        self.verbose=verbose
        self.N = len(self.dataframe)
        self.cross_val_split()
        self.normalizer = NORML()
        self.n_type=n_type
        if n_type is not None:
            self.normalize()

    def find_string_cols(self, df, ):
        rl = []
        for v in df.columns.tolist():
            #print(v)
            #print(df[v])
            if isinstance(df[v].values.tolist()[0], type(str(''))):
                rl.append(v)
        return rl

    def cross_val_split(self,):
        if self.train_test_split[0] == 0:
            self.Xtr = self.X
            self.Xts = self.X
            self.ytr = self.Y
            self.yts = self.Y
            return
        else:
            print(' **** the target is {} ****'.format(self.target))
            #print('the class 0 is {}'.format(self.classes[0]))
            if self.justify:
                all_classA = self.dataframe.loc[self.dataframe[self.target] == self.classes[0], :].values
                all_classB = self.dataframe.loc[self.dataframe[self.target] == self.classes[1], :].values
                # store position of target
                id = self.dataframe.columns.tolist().index(self.target)

                #print(len(all_classB[:, id].tolist()))

            #print(self.dataframe[self.target])
            #print(self.target)

                orig_size = self.dataframe.shape[0]
                percnt_a = all_classA.shape[0]/orig_size
                percnt_b = all_classB.shape[0]/orig_size

                tr_size = self.N * self.train_test_split[0]
                tr_size = int(np.around(tr_size, 0))
                tr_szA =int(np.around( all_classA.shape[0] * self.train_test_split[0], 0))
                tr_szB =int(np.around( all_classB.shape[0] * self.train_test_split[0], 0))

                # shuffle both
                # get half of both for the training set
                np.random.shuffle(all_classA)
                np.random.shuffle(all_classB)
                tr_side = np.append(all_classA[0:tr_szA], all_classB[0:tr_szB], axis=0)
                ts_side = np.append(all_classA[tr_szA:], all_classB[tr_szB:], axis=0)
                # now do one_last shuffle for the training side of things
                np.random.shuffle(tr_side)
                np.random.shuffle(ts_side)
                # now turn back into data frames to get x, and y for both
            #print('them cols\n')
            #print(self.columns)
            #print(len(self.columns))
                tr_side = pd.DataFrame(tr_side, columns=self.columns + [self.target])
                ts_side = pd.DataFrame(ts_side, columns=self.columns + [self.target])
                # now lets check those percentages again
                all_1 = tr_side.loc[tr_side[self.target] == self.classes[1], :]
                all_0 = tr_side.loc[tr_side[self.target] == self.classes[0], :]
                all_1s = ts_side.loc[ts_side[self.target] == self.classes[1], :]
                all_0s = ts_side.loc[ts_side[self.target] == self.classes[0], :]
            else:

                trsz = int(np.around(self.N * self.train_test_split[0], 0))
                print('\n\n\n\t\t\t***** Not justified N= {},tr= {} *****\n\n\n'.format(self.N, trsz))
                tssz = int(np.around(self.N - trsz, 0))
                tr_side = self.dataframe.iloc[0:trsz, :]
                ts_side = self.dataframe.iloc[trsz:, :]
            self.Xtr = pd.DataFrame(tr_side[self.columns])
            self.ytr = pd.DataFrame(tr_side[self.target])
            self.Xts = pd.DataFrame(ts_side[self.columns])
            self.yts = pd.DataFrame(ts_side[self.target])



            #print('Original there are {:.3f}% {}, and {:.3f}% {}'.format(percnt_a, self.classes[0], percnt_b, self.classes[1]))
            #print('TR there are {:.2f} of class {}'.format(all_0.shape[0]/tr_side.shape[0], self.classes[0]))
            #print('TR there are {:.2f} of class {}'.format(all_1.shape[0]/tr_side.shape[0], self.classes[1]))
            #print('TS there are {:.2f} of class {}'.format(all_0s.shape[0]/ts_side.shape[0], self.classes[0]))
            #print('TS there are {:.2f} of class {}'.format(all_1s.shape[0]/ts_side.shape[0], self.classes[1]))

    def normalize(self, n_type='minmax'):
        self.normalizer.set_type(n_type=n_type)
        self.normalizer.fit(self.Xts)
        self.Xtr = self.normalizer.transform(self.X)
        self.Xts = self.normalizer.transform(self.Xts)


class ML_Analyzer:
    """
        This class is the base class for any kind of classifier/regression with ML.
        Provides base templates for the functions needed to perform various ML tasks
    """

    # for basic LR: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # for cross validated LR (what this is set up for): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
    # for Randomforest classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    logistic_params = {
        'penalty': 'l2',
        'dual': False,  # if N(samples) > D(variables) leave false, otherwise set to true
        'tol': 0.0001,
        'Cs': 10.0,   # inverse of regularization strength, smaller #'s == stronger regularization
        'cv': None,   # cross validation folds (int) or generator (some method) see: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        'scoring': None,  # options for scorer: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': None,
        'random_state': None,
        'solver': 'lbfgs', # options: 'liblinear'(l1) (smaller sets), 'sag'(l2), 'saga'(l1), 'newton-cg'(l2), 'lbfgs'(l2)
        'max_iter': 100,
        'multi_class': 'auto',
        'l1_ratios': None,
        'refit': True,
        'verbose': 0,
        'warm_start': False,
        'n_jobs': None,
        'l1_ratio': None
    }
    randomforest_params = {
        'n_estimators': 100,
        'criterion': 'entropy',  #criterion{“gini”, “entropy”}, default=”gini”
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': None,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'class_weight': None,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

    randomforest_paramsR = {
        'n_estimators': 100,
        'criterion': 'mae',  # criterion{“gini”, “entropy”}, default=”gini”
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': None,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

    def __init__(self, clf_type, scaler_type=None, params=None, tr_split=(.5, ),
                 model_type='classifier'):
        self.clf_type=clf_type
        self.learner = None
        self.params = params
        self.model_type = model_type
        self.tr_split = tr_split     # training test split
        self.generate_classifier()
        self.scaler_type = scaler_type         # type of scaling to perform
        self.scaler = None
        self.scl_suffix = None
        self.feature_ranking_storage = {}
        self.generate_scaler()
        self.feature_importance = None
        self.features = None
        self.FI = None
        self.weights = None
        self.score_dict = dict({})

    def feature_importance(self):
        if self.RF_Check():
            return self.FI
        else:
            print('Only random forest classifiers have feature importance')
            print('This is a {} classifier'.format(self.clf_type))
            return {}

    def store_fi_ranking(self, fi, ):
        fts = list(fi.keys())
        self.feature_ranking_storage = {}
        for f in fts:
            self.feature_ranking_storage[f] = fts.index(f)
        self.feature_ranking_storage = sort_dict(self.feature_ranking_storage)
        return

    def RF_Check(self, verbose=False):
        if self.clf_type in ['rf', 'randomforest', 'random_forest', 'random forest']:
            return True
        return False
    def LR_Check(self, verbose=False):
        if self.clf_type in ['logr', 'logistic']:
            return True
        return False
    def fit(self, x, y, verbose=False, ):
        if isinstance(x, type(pd.DataFrame())):
            self.features = x.columns.tolist()
        self.learner.fit(x, y.values.flatten())
        if self.RF_Check():
            #howrint(dir(self.learner))
            self.feature_importance = self.learner.feature_importances_
            if self.features is not None:
                self.feature_importance = display_significance(self.feature_importance, self.features, verbose=verbose)
                self.FI = self.feature_importance
                self.store_fi_ranking(self.FI, )
        if self.LR_Check():
            self.feature_importance = self.learner.coef_[0]
            if self.features is not None:
                self.feature_importance = display_significance(self.feature_importance, self.features, verbose=verbose)
                self.FI = self.feature_importance
                self.weights = self.FI

    def score(self, X, y):
        ypred = self.learner.predict(X)
        if self.model_type == 'classification':
            acc = accuracy_score(y, ypred)
            sen = recall_score(y, ypred)
            prec = precision_score(y, ypred)
            R2 = explained_variance_score(y, ypred)
            self.score_dict = {
                'acc': acc,
                'sen': sen,
                'prec': prec,
                'R2': R2,
            }
            return ypred, acc, sen, prec, R2
        else:
            mae = mean_absolute_error(y, ypred)
            mse = mean_squared_error(y, ypred)
            cod = r2_score(y, ypred)
            R2 = explained_variance_score(y, ypred)
            self.score_dict = {
                        'mae': mae,
                        'mse': mse,
                        'cod': cod,
                        'R2': R2,
            }
            return ypred, mae, mse, cod, R2

    def generate_classifier(self):
        if self.clf_type in ['logr', 'logistic']:
            if self.params is None:
                self.params = self.logistic_params
            self.generate_logistic()
        elif self.clf_type in ['rf', 'randomforest', 'random_forest', 'random forest']:
            if self.params is None:
                if self.model_type == 'classifier':
                    self.params = self.randomforest_params
                else:
                    self.params = self.randomforest_paramsR
            self.generate_randomforest()

    def generate_randomforest(self):
        for param in self.params:
            if self.model_type == 'regression':
                self.randomforest_paramsR[param] = self.params[param]
            else:
                self.randomforest_params[param] = self.params[param]
        params = self.params
        #print(params)
        #print('the model again ----- {}'.format(self.model_type))
        if self.model_type == 'classifier' or self.model_type not in ['classifier', 'regression']:
            self.learner = RandomForestClassifier(n_estimators=params['n_estimators'],
                                              criterion=params['criterion'],
                                              max_depth=params['max_depth'],
                                              min_samples_split=params['min_samples_split'],
                                              min_samples_leaf=params['min_samples_leaf'],
                                              min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                              max_features=params['max_features'],
                                              max_leaf_nodes=params['max_leaf_nodes'],
                                              min_impurity_decrease=params['min_impurity_decrease'],
                                              min_impurity_split=params['min_impurity_split'],
                                              bootstrap=params['bootstrap'],
                                              oob_score=params['oob_score'],
                                              n_jobs=params['n_jobs'],
                                              random_state=params['random_state'],
                                              verbose=params['verbose'],
                                              warm_start=params['warm_start'],
                                              class_weight=params['class_weight'],
                                              ccp_alpha=params['ccp_alpha'],
                                              max_samples=params['max_samples'])
        elif self.model_type == 'regression':
            #print('Generating RFR')
            self.learner = RandomForestRegressor(n_estimators=params['n_estimators'],
                                              criterion=params['criterion'],
                                              max_depth=params['max_depth'],
                                              min_samples_split=params['min_samples_split'],
                                              min_samples_leaf=params['min_samples_leaf'],
                                              min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                              max_features=params['max_features'],
                                              max_leaf_nodes=params['max_leaf_nodes'],
                                              min_impurity_decrease=params['min_impurity_decrease'],
                                              min_impurity_split=params['min_impurity_split'],
                                              bootstrap=params['bootstrap'],
                                              oob_score=params['oob_score'],
                                              n_jobs=params['n_jobs'],
                                              random_state=params['random_state'],
                                              verbose=params['verbose'],
                                              warm_start=params['warm_start'],
                                              ccp_alpha=params['ccp_alpha'],
                                              max_samples=params['max_samples'])
        return

    def generate_logistic(self):
        for param in self.params:
            self.logistic_params[param] = self.params[param]
        params = self.params
        self.learner = LogisticRegressionCV(Cs=params['Cs'],
                                            fit_intercept=params['fit_intercept'],
                                            cv=params['cv'],
                                            dual=params['dual'],
                                            penalty=params['penalty'],
                                            scoring=params['scoring'],
                                            solver=params['solver'],
                                            tol=params['tol'],
                                            max_iter=params['max_iter'],
                                            class_weight=params['class_weight'],
                                            n_jobs=params['n_jobs'],
                                            verbose=params['verbose'],
                                            refit=params['refit'],
                                            intercept_scaling=params['intercept_scaling'],
                                            multi_class=params['multi_class'],
                                            random_state=params['random_state'],
                                            l1_ratios=params['l1_ratios'])

        return

    def generate_scaler(self):
        if self.scaler_type in ['minmax', ]:
            self.scaler = MinMaxScaler()
            self.scl_suffix = '_nrm'
        elif self.scaler_type in ['Z', 'standard']:
            self.scaler = StandardScaler()
            self.scl_suffix = '_Z'


class Logistic_Regression_Analzer(ML_Analyzer):
    def __init__(self, clf_type='logistic', scaler_type=None, params=None, tr_split=(.5, )):
        super().__init__(clf_type, scaler_type=scaler_type, params=params, tr_split=tr_split)

class RandomForest_Analzer(ML_Analyzer):
    def __init__(self, clf_type='randomforest', scaler_type=None, params=None, tr_split=(.5, ),model_type='classifier'):
        super().__init__(clf_type, scaler_type=scaler_type, params=params, tr_split=tr_split, model_type=model_type)



# =========================================================================
# =========================================================================
#                   TODO: Dimension Reduction tools
# =========================================================================
# =========================================================================

class DimensionReducer():
    def __init__(self):
        self.type=None
        self.class_splits=None
        self.classes=None
        self.class_means=None
        self.data_means=None
        self.data_std = None
        self.class_cov=None
        self.class_cov_inv=None
        self.class_cov_det=None
        self.class_priors=None
        self.class_counts=None
        self.eig_vec = None
        self.eig_vals = None
        self.pval = None
        self.W = None
        self.WT = None
        self.WT2 = None
        self.k=None
        self.k_90 = None
        self.s = None
        self.vh = None
        self.i_l = list()
        self.dr_type = None
        self.y2 = None
        self.x2 = None
        self.x1 = None
        self.y1 = None
        self.N = None
        self.z, self.one = list(), list()

    def FLDA(self, df, dftr, y, classes=(0,1), class_label='type'):
        y = pd.DataFrame(y, columns=[class_label])
        c1 = dftr.loc[y[class_label] == classes[0], :]
        n1 = len(c1)
        c2 = dftr.loc[y[class_label] == classes[1], :]
        n2 = len(c2)
        #print('There are {0} negative and {1} positive samples'.format(n1, n2))
        Sw_inv = np.linalg.inv((n1-1)*c1.cov() + (n2-1)*c2.cov())
        #print(Sw_inv.shape)
        #print(c1.mean().shape)
        #w = np.dot(Sw_inv, np.dot((c1.mean() - c2.mean()), (c1.mean()-c2.mean()).transpose()))
        w = np.dot(Sw_inv, (c1.values.mean(axis=0) - c2.values.mean(axis=0)))
        #print('w', w.shape)
        #print('df', df.shape)
        return np.dot(df,w)
    def fld_fit(self, cmodel):
        self.dr_type = 'fld'
        self.attribs = cmodel.attribs
        self.class_splits = cmodel.class_splits
        self.class_counts = cmodel.class_counts
        self.classes = cmodel.classes
        self.class_means = cmodel.class_means
        self.data_means = cmodel.model_mean
        self.data_std = cmodel.model_std
        self.class_cov = cmodel.class_cov
        self.class_cov_inv = cmodel.class_cov_inv
        self.class_cov_det = cmodel.class_cov_det
        self.class_priors = cmodel.class_priors
        self.N=None
        self.kmm=None
        self.Calculate_W_FLD()
    def pca_fit(self, X):
        self.eig_vals, self.eig_vec = np.linalg.eig(X.cov())
        #print('eigvec', self.eig_vec)
        self.N = len(X)
        print('eigvals', self.eig_vals)
    def svd_w_np(self, X):
        u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
        self.s = s
        self.vh = vh
        self.N = len(X)
        self.d = X.shape[1]
        return
    def svd_pov(self, s, accuracy=.90, verbose=False, pov_plot=False, show_now=False):
        sum_s = sum(s.tolist())
        ss = s**2
        sum_ss = sum(ss.tolist())
        self.prop_list = list()
        found = False
        k = 0
        x1, y1, x2, y2, = 0, 0, 0, 0
        p_l, i_l = 0, 0
        found = False
        self.prop_list.append(0)
        self.i_l.append(0)
        for i in range(1, len(ss)+1):
            perct = sum(ss[0:i]) / sum_ss
            # perct = sum(s[0:i]) / sum_s
            if np.around(perct, 2) >= accuracy and not found:
                self.x1 = i
                self.y1 = perct
                found = True
            self.prop_list.append(perct)
            self.i_l.append(i)
        self.single_vals = np.arange(1, self.N + 1)
        if pov_plot:
            plt.figure()
            plt.plot(self.i_l, self.prop_list)
            plt.scatter(self.x1, self.y1, c='r', marker='o', label='Point at {:.1f}% accuracy'.format(self.y1*100))
            plt.title('Proportion of Variance vs. Number of Eigen Values\n{:d} required for {:.2f}'.format(self.x1, self.y1*100))
            plt.legend()
            plt.xlabel('Number of Eigen values')
            plt.ylabel('Proportion of Variance')
            if show_now:
                plt.show()
        return self.x1 + 1
    def svd_fit(self,X, vh=None, k=None, get_pov=True, pov_thresh=.90, verbose=False, plot=False, usek=False,
                gen_plot=True, y=None):
        if vh is None:
            self.svd_w_np(X)
            u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
            if get_pov:
                print('getting pov')
                self.kmm =  self.svd_pov(s, accuracy=pov_thresh, verbose=verbose, pov_plot=plot, show_now=True)
            if usek:
                k = self.kmm
        self.N = len(X)
        self.data_means = X.mean(axis=0).values.flatten()
        #print('data means', self.data_means)
        print('vector shape', vh.shape)
        #vt = np.transpose(self.vh)
        vt = np.transpose(self.vh)
        # grab the first k principle components
        if k is not None:
            self.W = vt[:, 0:k]
            self.k = k
        else:
            self.W = vt[:, :]
            self.k = len(X)
        self.WT = np.transpose(self.W)

        # grab the first two principle components
        W2 = vt[:, 0:2]
        W3 = vt[:, 0:3]
        self.WT2 = np.transpose(W2)
        self.WT3 = np.transpose(W3)
        if gen_plot:
            # get 0's and 1's
            for row, adp in zip(self.WT2, y):
                if adp == 0:
                    self.z.append(row)
                else:
                    self.one.append(row)
    def svd_transform(self, X, treD=False):
        z_array = list()
        z2_array = list()
        z3_array = list()
        for row in X.values:
            #print('row',row)
            #print('data means')
            #print(self.data_means)
            c_x = row - self.data_means
            z_array.append(np.dot(self.WT, c_x))
            z2_array.append(np.dot(self.WT2, c_x))
            if treD:
                z3_array.append(np.dot(self.WT3, c_x))
        Z = np.array(z_array, dtype=np.float)
        Z2 = np.array(z2_array, dtype=np.float)
        if treD:
            Z3 = np.array(z3_array, dtype=np.float)
            return Z, Z2, Z3
        return Z, Z2
    def Calculate_W_FLD(self, ):
        c1 = self.class_splits[0]
        n1 = self.class_counts[0]
        c2 = self.class_splits[1]
        n2 = self.class_counts[1]
        # TODO: evalute below to see if needed or not
        if False and type(c1) != type(pd.DataFrame({0:0})):
            c1 = pd.DataFrame(c1).values
            c2 = pd.DataFrame(c2).values
        Sw_inv = np.linalg.inv((n1 - 1) * c1.cov() + (n2 - 1) * c2.cov())
        self.W_fld = np.dot(Sw_inv, (c1.values.mean(axis=0) - c2.values.mean(axis=0)))

    def pca_transform(self, X, p=None):
        if p is None:
            return self.convert_basis(X, self.eig_vec)
        else:
            return self.convert_basis(X, self.eig_vec[0:p])

    def FLD(self, X):
        return pd.DataFrame(np.dot(X,self.W_fld))

    def PCA(self, df, pov, eig_vec=None, m=None, verbose=False, ret_level=0,
            pov_plot=False, show_now=False):
        # if not given eigen values calculate them based on the
        # desired proportion of variance (pov) covered
        if eig_vec is None:
            #print(df.cov())
            eig_vals, eig_vec = np.linalg.eig(df.cov())
            if m is None:
                pov_l, m = self.calculate_p(eig_vals=eig_vals, pov=pov, verbose=verbose, show_now=show_now,
                                            pov_plot=pov_plot)
                print('The Number of eigenvectors to cover {:.2f} of the variance is {:d}'.format(100*pov, m))
            else:
                pov_l, cm = self.calculate_p(eig_vals=eig_vals, pov=pov, verbose=verbose, show_now=show_now,
                                             pov_plot=pov_plot)
                print('The number of eigenvectors is set to {:d}'.format(m))
            #print(eig_vec)
            eig_vec = eig_vec[0:m]
            # now perform transform
            y = self.convert_basis(df, pd.DataFrame(eig_vec))
            if ret_level == 0:
                return y
            elif ret_level == 2:
                return pov_l, m, eig_vec, pd.DataFrame(y)
            elif ret_level == 1:
                return eig_vec, y

    def calculate_p(self, pov, eig_vals=None,  verbose=False, pov_plot=False, show_now=False):
        #print('calculating k')
        # calculate total sum
        if eig_vals is None:
            eig_vals = self.eig_vals
        s_m = sum(eig_vals)
        if verbose:
            print('The sum of the eigen values is {0}'.format(s_m))
            print('The length of the eigen values vector is {0}'.format(len(eig_vals)))
            print('here it is')
            print(eig_vals)
        # now go through to find your required k for the
        # desired Proportion of Variance (pov)
        pov_l, cpov, csum, k, found = [], 0, 0, 0, False
        pov_found = 0
        for v in range(len(eig_vals)):
            csum += np.around(eig_vals[v], 2)
            pov_l.append(np.around(csum/s_m, 2))
            if verbose:
                print('The sum at {:d} is {:.2f}, pov {:.2f}'.format(v, csum, pov_l[v]))
            if pov_l[-1] >= pov and not found:
                k = v+1
                pov_found = pov_l[-1]
                print(k, pov_found)
                found = True
        if pov_plot:
            plt.figure()
            plt.plot(list(range(1, len(eig_vals)+1)), pov_l)
            plt.scatter(k, pov_found, c='r', marker='o', label='Point at {:.1f}% accuracy'.format(pov_found*100))
            plt.title('Proportion of Variance vs. Number of Eigen Values')
            plt.legend()
            plt.xlabel('Number of Eigen values')
            plt.ylabel('Proportion of Variance')
            if show_now:
                plt.show()

        self.pov_l = pov_l
        self.k = k
        return pov_l, k

    def convert_basis(self, df, new_basis):
        return np.dot(df, new_basis.transpose())

    def PCA_Eig(self, X, class_means):
        return 1

    def PCA_SVD(self, X):
        return 1

# =========================================================================
# =========================================================================
#                   TODO: Learners
# =========================================================================
# =========================================================================

def epsilon(emax, emin, k, kmax):
    return emax * ((emin/emax)**(min(k, kmax)/kmax))

class Learner(ABC):
    """Template class for a learning machine"""
    def __init__(self,):
        pass
    def finish_init(self):
        pass
    def fit(self, cmodel):
        pass
    def predict(self, X):
        pass
    def score(self, X, Y):
        pass

class bayes_classifiers(Learner):
    def __init__(self, cmodel=None, df_list=None, case=1):
        super().__init__()
        self.fit(cmodel=cmodel)
        self.case = case
        if self.case == 1:
            self.func = 'euclid'
        elif self.case == 2:
            self.func = 'mahala'
        if self.case == 1:
            self.func = 'euclid'
        elif self.case == 3:
            self.func = 'quadratic'
    def fit(self, cmodel):
        self.cmodel = cmodel

    def bayes_classifier_model_finder(self, dfx, dfy, case=1, func='euclid', scale= .1, verbose=False,
                                      priors=()):
        #n, p = list(), list()
        #f, b = .10, .90

        #while f <= b:
        #    n.append(b)
        #    p.append(f)
        #    f += scale
        #    b -= scale
        #back = n[0:]
        #rback = n[0:]
        #rback.reverse()
        #ford = p[0:-1]
        #rford = ford[0:]
        #rford.reverse()
        #pos = ford + rback
        #neg = back + rford

        if len(priors) == 0:
            pos, neg = self.generate_priors(scale=scale)
        else:
            if scale is not None:
                pos, neg = self.generate_priors(scale=scale, prior1=priors[0], prior2=priors[1])
            else:
                pos, neg = list([priors[0]]), list([priors[1]])
        #print(pos)
        #print(neg)
        best_acc, best_scr, best_posnegs, scr = 0,  None, None, 0
        pr_l1, pr_l2, accuracy , sens, spec= list(), list(), list(), list(), list()
        best_pr = [0, 0]
        for ps, ng in zip(pos, neg):
            pr = [ps, ng]
            #print(pr[0] + pr[1])
            acc, scr, posnegs = self.bayes_classifier_predict_and_score(dfx, dfy, case=case, priors=pr, func=func, verbose=verbose)
            accuracy.append(acc)
            pr_l1.append(pr[0])
            pr_l2.append(pr[1])
            sens.append(posnegs['sen'])
            spec.append(posnegs['spe'])
            if acc > best_acc:
                best_acc = acc
                best_scr = scr
                best_posnegs = posnegs
                best_pr[0] = pr[0]
                best_pr[1] = pr[1]

        result_dic = {'best_accuracy':best_acc,
                      'best_scores':best_scr,
                      'best_postnegs':best_posnegs,
                      'pr_l1':pr_l1,
                      'pr_l2':pr_l2,
                      'sens':sens,
                      'spec':spec,
                      'accuracy_list':accuracy,
                      'best_priors': best_pr}

        #return best_acc, best_scr, best_posnegs, pr_l1, pr_l2, sens, spec, accuracy, best_pr
        return result_dic

    def bayes_classifier_model_finderB(self, dfx, dfy, case=1, func='euclid', scale= .1, verbose=False,
                                      priors=()):
        #n, p = list(), list()
        #f, b = .10, .90

        #while f <= b:
        #    n.append(b)
        #    p.append(f)
        #    f += scale
        #    b -= scale
        #back = n[0:]
        #rback = n[0:]
        #rback.reverse()
        #ford = p[0:-1]
        #rford = ford[0:]
        #rford.reverse()
        #pos = ford + rback
        #neg = back + rford

        if len(priors) == 0:
            pos, neg = self.generate_priors(scale=scale)
        else:
            if scale is not None:
                pos, neg = self.generate_priors(scale=scale, prior1=priors[0], prior2=priors[1])
            else:
                pos, neg = list([priors[0]]), list([priors[1]])
        print(pos)
        print(neg)
        best_acc, best_scr, best_posnegs, scr = 0,  None, None, 0
        pr_l1, pr_l2, accuracy , sens, spec= list(), list(), list(), list(), list()
        best_pr = [0, 0]
        for ps, ng in zip(pos, neg):
            pr = [ps, ng]
            #print(pr[0] + pr[1])
            acc, scr, posnegs = self.bayes_classifier_predict_and_scoreB(dfx, dfy, case=case, priors=pr, func=func, verbose=verbose)
            accuracy.append(acc)
            pr_l1.append(pr[0])
            pr_l2.append(pr[1])
            sens.append(posnegs['sen'])
            spec.append(posnegs['spe'])
            if acc > best_acc:
                best_acc = acc
                best_scr = scr
                best_posnegs = posnegs
                best_pr[0] = pr[0]
                best_pr[1] = pr[1]

        result_dic = {'best_accuracy': best_acc,
                      'best_scores': best_scr,
                      'best_postnegs': best_posnegs,
                      'pr_l1': pr_l1,
                      'pr_l2': pr_l2,
                      'sens': sens,
                      'spec': spec,
                      'accuracy_list': accuracy,
                      'best_priors': best_pr}

        # return best_acc, best_scr, best_posnegs, pr_l1, pr_l2, sens, spec, accuracy, best_pr
        return result_dic

    def bayes_classifier_predict(self, dfx, case=1, verbose=False, priors = [1,1], func='euclid'):
        ypred = list()
        case = self.case
        func = self.func
        # figure out the priors situation
        if priors is None:
            prior1 = self.cmodel.class_priors[0]
            prior2 = self.cmodel.class_priors[1]
        else:
            prior1 = priors[0]
            prior2 = priors[1]
        #print('====================================>   Priors: ', prior1, prior2)

        # figure out what discriminant function to use
        if case == 1:
            func = 'euclid'
            #cov = self.gauss_params['std'] ** 2
            cov = self.cmodel.model_std  ** 2
            print('cov', cov)
        elif case == 2:
            func = 'mahala'
            cov = self.cmodel.model_cov
        elif case == 3:
            func = 'quadratic'
            cov = [self.cmodel.class_cov[0], self.cmodel.class_cov[1]]

        #mu1 = self.gauss_params['mu_c1']
        #mu1 = self.gauss_params['mu_c1']
        #mu1 = self.Cmu_array[0]
        #mu2 = self.Cmu_array[1]
        mu1 = self.cmodel.class_means[0]
        mu2 = self.cmodel.class_means[1]
        #print(func)
        # make some predictions
        for xi in dfx.values:
            if case != 3:
                pc1 = self.discriminate_function(xi, mu1, cov, prior1, func=func)
                pc2 = self.discriminate_function(xi, mu2, cov, prior2, func=func)
            else:
                #print(func)
                pc1 = self.discriminate_function(xi, mu1, cov[0], prior1, func=func)
                pc2 = self.discriminate_function(xi, mu2, cov[1], prior2, func=func)
            if verbose:
                print('pc1',pc1)
                print('pc2',pc2)
            if pc1 > pc2:
                ypred.append(0)
            else:
                ypred.append(1)
        return ypred

    def bayes_classifier_predict_and_scoreB(self, dfx, dfy, case=1, priors=[1,1], func='euclid', verbose=False):
        case = self.case
        func = self.func
        ypred = self.bayes_classifier_predict(dfx, case=case, priors=priors, func=func, verbose=verbose)
        return self.bayes_classifier_score(dfy, ypred)

    def bayes_classifier_predict_and_score(self, dfx, dfy, case=1, priors=[1,1], func='euclid', verbose=False):
        case = self.case
        func = self.func
        ypred = self.bayes_classifier_predict(dfx, case=case, priors=priors, func=func, verbose=verbose)
        return self.bayes_classifier_score(dfy, ypred)

    def bayes_classifier_score(self, yactual, ypred, vals = [0,1]):
        return bi_score(ypred, yactual, vals, classes=vals)

    def generate_priors(self, scale, prior1=None, prior2=None):
        if prior1 is not None and prior2 is not None:
            # set up prior1 side
            if prior1 < prior2:
                l1 = list([1])
                l2 = list([.0001])
                while l1[-1] - scale >= prior1:
                    l1.append(l1[-1] - scale)
                    l2.append(1 - l1[-1])
                if l1[-1] - prior1 < 0:
                    l1[-1] = prior1
                    l2[-1] = 1 - prior1
            else:
                l1 = list([.0001])
                l2 = list([1])
                while l2[-1] - scale >= prior2:
                    l2.append(l2[-1] - scale)
                    l1.append(1 - l2[-1])
                if l2[-1] - prior2 < 0:
                    l2[-1] = prior2
                    l1[-1] = 1 - prior2
            return l1, l2

        l1 = list([0.001])
        l2 = list([1])
        while np.around(l1[-1] + scale, 3) < 1:
            l1.append(np.around(l1[-1] + scale, 3))
            l2.append(np.around(1 - l1[-1], 3))
        return l1, l2

    def predict(self, X, case=1, num_cls = 2, priors=(1,1)):
        return self.bayes_classifier_predict(X, case=1, verbose=False, priors=priors)
        #if num_cls == 2:
        #    return self.generate_predictions_bi(X, case=case)

    def score(self, Ya, Yp):
        pass

    def dim_reduce(self, type='', attribs=()):
        pass

    def euclidian_disc(self, mu, x, cov, prior):
        return (-np.dot(x.transpose(), np.dot(mu, x))/cov**2) + np.log(prior)

    def mahalanobis_disc(self, mu, x, cov_inv, prior):
        return -np.dot(x.transpose(), np.dot(mu, x)) + np.log(prior)

    def quadratic_disc(self, mu, x, cov_inv, cov_det, prior):
        return -np.dot(x.transpose(), np.dot(mu, x)) + np.log(prior)

    def min_euclid(self, mean_ib, xvec, sig1, prior):
        return (np.dot(mean_ib.T, xvec) / sig1) - (np.dot(mean_ib.T, mean_ib) / (sig1 * 2)) + np.log(prior)
        #return -(np.sqrt((np.linalg.norm(xvec-mean_ib))))/(2*sig1) + np.log(prior)

    def min_mahalanobis(self,mu,x,siginv,prior):
        return np.dot(mu.T,np.dot(siginv.T, x)) - (.5 * np.dot(mu.T, np.dot(siginv, mu))) + np.log(prior)
    def quadratic_machine(self, x, mu, siginv, detsig, prior):
        return (-.5 * np.dot(x.T, np.dot(siginv, x))) + np.dot(mu.T, np.dot(siginv.T, x)) - (.5*np.dot(mu.T, np.dot(siginv, mu))) - (.5*np.log(detsig))+np.log(prior)
        #return (-.5 * np.dot(x.T, np.dot(siginv, x))) + np.dot(np.dot(siginv, mu).T, x) - (.5*np.dot(mu.T, np.dot(siginv, mu))) - (.5*np.log(detsig))+np.log(prior)

    def generate_predictions_bi(self,X, case=1):
        y = list()
        for x in X:
            # get the posterior probability of
            # and set the class as the MPP
            c1 = self.case_picker(X, case, 0)
            c2 = self.case_picker(X, case, 1)
            if c1 > c2:
                y.append(0)
            else:
                y.append(1)
        return y

    def case_picker(self, X, case, class_val):
        case = self.case
        if case == 1:
            return self.min_euclid(self.cmodel.class_means[class_val], X, self.cmodel.model_std**2,
                                   self.cmodel.class_priors[class_val])
            #return self.euclidian_disc(self.cmodel.class_means[class_val], X, self.cmodel.,
            #                           self.cmodel.class_priors[class_val])
        elif case == 2:
            return self.mahalanobis_disc(self.cmodel.class_means[class_val], X, self.cmodel.model_cov_inv,
                                       self.cmodel.class_priors[class_val])
        elif case == 3:
            return self.quadratic_disc(self.cmodel.class_means[class_val], X, self.cmodel.class_cov_inv[class_val],
                                         self.cmodel.class_cov_det[class_val], self.cmodel.class_priors[class_val])

    def discriminate_function(self, df, mu, cov, prior, func='euclid', verbose=False):
        func = self.func
        if func.lower() == 'euclid':
            #print('euclid')
            if verbose:
                print('X:\n', df)
                print('mu:\n', mu)
                print('std:', cov)
            return self.min_euclid(mu, df, cov, prior)
        elif func.lower() == 'mahala':
            return self.min_mahalanobis(mu, df, np.linalg.inv(cov), prior)
            #print('mahala')
        elif func.lower() == 'quadratic':
            #print('quad')
            return self.quadratic_machine(df, mu, np.linalg.inv(cov), np.linalg.det(cov), prior)

class clusters():
    """Represents a group of clusters"""
    def __init__(self, k, method='kmeans', init='random', df=None, distance_metric='dmin',
                 distance_calc='euclid', verbose=True, distance_type='city_block'):
        self.methods = ['kmu', 'wta', 'kohonen']
        self.init_types = ['random', 'random_sample', 'normal']
        self.k = k                                                                  # desired number of clusters
        self.df = df                                                                # the data frame I was given
        self.del_thrsh = .09
        self.dist_type = distance_type
        self.distance_metric = distance_metric                                      # what type of distance metric used
        self.distance_calc = distance_calc                                          # how to calculate the distance
        #self.data = self.df.values                                                 # the numpy array of my data
        self.size = df.shape[0]                                                     # the number of samples in the data
        self.dimen = df.shape[1]                                                    # the number of features
        self.method = method                                                        # the clustering method
        print('method:', method)
        self.top_grid = list()
        self.epochs = 0
        self.emax = 1
        self.emin = .0001
        self.kmax =100
        self.time_taken = 0
        self.std = int(np.around(df.values.std(axis=0).mean(),0))
        self.mu = int(np.around(df.values.mean(axis=0).mean(),0))
        print('std ', self.std)
        print('mu ', self.mu)
        self.init = init                                                            # the method of initializing clusters
        self.dist_LUT = None
        #if distance_metric in ['dmin', 'dmax']:
        #    print('dminmax')
        #    self.dist_LUT = self.calculate_distance_LUT(df.values)                               # generate look up table of distances
        self.my_clusters = None
        self.my_clusters = self.check_method(df.values)                             # the list of clusters initialized
        if verbose:
            print('There are {:d} clusters to start'.format(self.check_size()))

    def set_threshold(self, ):
        if self.distance_metric in ['dmax', 'dmean']:
            return -9999
        elif self.distance_metric == 'dmin':
            return 9999

    def perform_dist_test(self, threshold):
        if self.distance_metric == 'dmin':
            pass
    def calculate_needed_dist(self, Apt, Bpts):
        na_row = self.dist_LUT[Apt]
        # if we are looking for dmin (minimum distanct between clusters)
        if self.distance_metric == 'dmin':
            bpt, distance = get_select_min_idx(na_row, Bpts)
            return distance
        # if we are looking for dmin (minimum distanct between clusters)
        elif self.distance_metric == 'dmax':
            bpt, distance = get_select_max_idx(na_row, Bpts)
            return distance
    def update_means(self, cls, data):
        for n in cls:
            if len(cls[n][1]) > 0:
                cls[n][0] = np.array(np.around(data[cls[n][1]].mean(axis=0), 0), dtype=np.int)
        return cls
    def perform_epoch(self):
        threshold = self.set_threshold()
        clusterA, clusterB = 0, 0
        # for each cluster look up the  distance between it's inhabitants
        # and all other inhabitants
        for c1 in range(len(self.my_clusters)-1):
            for c2 in range(c1 + 1, len(self.my_clusters)):
                # grab the two cluster list of point the cover
                c1_pts = self.my_clusters[c1].inhabitants
                c2_pts = self.my_clusters[c2].inhabitants

                #for p1 in self.my_clusters[c1].inhabitants:
                # go through cluster 1's points and look at the distance
                # between each of those, and each of the ones in the
                # current other cluster
                for p1 in c1_pts:
                    better, distance, = self.calculate_needed_dist(threshold=threshold, Apt=p1, Bpts = c2_pts)


                    # grab the 2nd clusters points
                    #for p2 in self.my_clusters[c2].inhabitants:
                    #    # compare to current threshold and if it is better
                    #    if self.dist_LUT[p1][p2] > threshold:
                    #        pass

        # and every other clusters inhabitants
        # based on whether we are looking at d max
        # dmin or dmean keep track of the shortest one
        # and whitch two clusters this involves
        # once done merge the two with min distance and repeat
        # until desired number of clusters is found
    def check_size(self):
        return len(self.my_clusters)
    def merge(self, c1i, c2i, verbose=True):
        """Hopefully will merge the two clusters"""
        c1 = self.my_clusters[c1i]
        c2 = self.my_clusters[c2i]
        # get the average for the new cluster
        self.my_clusters[c1i].value = np.stack(c1.value, c2.value).mean(axis=0)
        c1.inhabitants += c2.inhabitants
        if verbose:
            print('the merged inhabitants are ')
            print(c1.inhabitants)
            quit(-745)
        return
    def calculate_distance_LUT(self, data):
        """Will Create a look up table for the distance
           from each point to every other thing
        """
        tstart = time.time()
        adj = list(([[0]*self.size]*self.size))
        for row in range(self.size):
            for col in range(self.size):
                if row == col:
                    if self.distance_metric == 'dmin':
                        adj[row][col] = 9000
                    elif self.distance_metric == 'dmax':
                        adj[row][col] = -9000
                else:
                    if self.dist_type == 'city_block':
                        #print('city block')
                        adj[row][col] = np.linalg.norm(data[row]-data[col])
                     # dc[i2] = np.linalg.norm(self.data[i1] - self.data[i2])
            # rd[i1] = sort_dict(dc)
        #pd.DataFrame(adj[0:len(self.size)/4], dtype=np.int).to_excel('The_LUT.xlsx')
        print('Making the LUT took {}'.format(time.time()-tstart))
        return np.array(adj, dtype=int)
    def calculate_cluster_diffs(self, rl):
        for cls in range(len(rl)-1):
            for cls2 in range(cls+1, len(rl)):
                dis = np.linalg.norm((rl[cls].value - rl[cls2]))
                rl[cls].cluster_dist[cls2] = dis
                rl[cls2].cluster_dist[cls] = dis
        # sort the dictionary of distances by value
        for cls in range(len(rl)):
            rl[cls].cluster_dict = sort_dict(rl[cls].cluster_dict)
        return rl
    def dmin_merge(self):
        pass
    def dmax_merge(self):
        pass
    def dmean_merge(self):
        pass
    def merge_clusters(self):
        if self.distance_metric == 'dmin':
            self.dmin_merge()
        elif self.distance_metric == 'dmax':
            self.dmax_merge()
        elif self.distance_metric == 'dmean':
            self.dmean_merge()
    def epsilon(self, emax, emin, k, kmax):
        return emax * ((emin/emax)**(k/kmax))

    def wta_update_cls(self, cmean, X, verbose=False, epsln=.001):
        return cmean + epsln*(X - cmean)
    def wta_init_run(self, data, change_threshold=0.09):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        change_threshold = self.del_thrsh
        epsln = .1
        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(get_rounded_int_array(gaussrndm.rvs(3)))
            cls[l].append(list())
        #cls = self.update_means(cls, data)
        tot = 0
        #for c in cls:
        #    print(cls[c])
        #    print(c)
        #    tot += len(cls[c][1])

        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        while change:
            change = False
            change_cnt = 0
            # for each sample pixel
            # find its nearest mean and put in its
            # cluster and adjust that cluster
            # mean toward the new point
            for sample in range(len(data)):
                dis = 999999
                best = None
                cnt = 0
                # go through all clusters
                for i in cls:
                    # if using dmin or max
                    if self.distance_metric in ['dmin', 'dmax']:
                        if len(cls[i][1]) == 0:
                            cdis = np.linalg.norm(cls[i][0] - data[sample])
                        elif self.distance_metric == 'dmin':
                            if cnt == 0:
                                print('dmin')
                            tmpd = 99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample]-data[pt])
                                    if dp < tmpd and dp != 9999:
                                        #print(dp, tmpd)
                                        tmpd = dp
                                        best = i
                            cdis = tmpd
                            #if cnt == 0:
                            #    print('the min dis is {}'.format(cdis))
                            cnt += 1
                        elif self.distance_metric == 'dmax':
                            tmpd = -99999
                            for pt in cls[i][1]:
                                dp = np.linalg.norm(data[sample]-data[pt])
                                if dp > tmpd:
                                    tmpd = dp
                            cdis = tmpd
                    elif self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        elif np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean().values
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        #cdis = mahalanobis_distance(data[sample], cls[i][0])
                    cnt += 1
                    #cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    #print('evaluating {}'.format(cdis))
                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in this cluster
                # keep going
                if sample in cls[best][1]:
                    continue
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                            break
                    cls[best][1].append(sample)
                    # now update the center
                    cls[best][0] = self.wta_update_cls(cls[best][0], data[sample], epsln=epsln)
            # once we are done with this run adjust means
            # cls = self.update_means(cls, data)
            # at end of loop see what % of points changed
            # if less than threshold stop
            if self.epochs > 0 and self.epochs%1 == 0:
                #epsln = epsln *.1
                epsln = self.epsilon(emax=.1, emin=.0001, k=self.epochs, kmax=30)
                print('-----------------------------epsilon', epsln)
            if (change_cnt/self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt/self.size))
            elif self.epochs%50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt/self.size), epsln))

            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def kmean_init_run(self, data, change_threshold=.001):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        #change_threshold = self.del_thrsh
        rdch = np.random.choice(range(self.size), self.size, replace=False)
        start = 0
        end = int(self.size/self.k)
        step = int(self.size/self.k)
        #print('step 886', step)
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(get_rounded_int_array(gaussrndm.rvs(3)))
            cls[l].append(list())
            for i in range(start, end):
                cls[l][1].append(rdch[i])
            start = end
            end = min(end + step, self.size)
        # initialize the means based on whats in the
        cls = self.update_means(cls, data)
        #tot = 0
        #for c in cls:
        #    #print(cls[c])
        #    #print(c)
        #    tot += len(cls[c][1])
        #print('total', tot)
        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        self.epochs = 0
        print('Starting the while loop')
        epsln = .1
        while change:
            change = False
            change_cnt = 0
            # perform the epoch
            #  for every sample
            for sample in range(len(data)):
                dis = 999999
                best = None
                change_cnt = 0
                # go through current means
                # and find the closest
                for i in cls:
                    if self.distance_metric in ['dmin', 'dmax']:
                        #print('dmin or max')
                        if len(cls[i][1]) == 0:
                            cdis = np.linalg.norm(cls[i][0] - data[sample])
                        elif self.distance_metric == 'dmin':
                            tmpd = 99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample] - data[pt])
                                    if dp < tmpd:
                                        tmpd = dp
                            cdis = tmpd
                        elif self.distance_metric == 'dmax':
                            tmpd = -99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample] - data[pt])
                                    if dp > tmpd:
                                        tmpd = dp
                            cdis = tmpd
                    elif self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if len(cls[i][1]) <= 1 and np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        elif len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean()
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            #cdis = mahalanobis_distance(data[sample], cls[i][0], cov)


                        if self.epochs < 5:
                            print('my mahala in kmean')
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).cov().values
                            print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in the closest cluster
                # stay there and go to next sample
                if sample in cls[best][1]:
                    continue
                # otherwise put the sample in it's
                # closest cluster after removing
                # it from its current one then
                # set that a change occurred
                # and keep track of how many
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                    cls[best][1].append(sample)
                    cls = self.update_means(cls, data)

            if self.epochs > 0 and self.epochs % 1 == 0:
                #epsln = epsln * .1
                epsln = self.epsilon(.1, .0001, self.epochs, 40)
                print('-----------------------------epsilon', epsln)
            if (change_cnt / self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt / self.size))
            elif self.epochs % 50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt / self.size), epsln))
            # once we are done with this run adjust means
            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        print('it took {} epochs'.format(self.epochs))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def phi(self, coord1, coordwinner, sig=1):
        if coord1 in [0,self.k-1] and coordwinner in [0, self.k-1] and coord1 != coordwinner:
            coord1, coordwinner = 1, 0
        return np.exp(-1*((((coord1 - coordwinner)**2)/(2*sig**2))))
    def kohonen_update_cls(self, cmeans, X, winner, verbose=False, epsln=.01, alpha=.0001):
        for i in range(len(cmeans)):
            cmeans[i][0] = cmeans[i][0] + epsln*self.phi(i, winner)*(X - cmeans[i][0])
        return cmeans
    def kohonen_init_run(self, data, change_threshold=.01):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        #change_threshold = self.del_thrsh
        epsln = .1
        tmp_dict = dict()
        vecs, dists = list(), list()
        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            vecs.append(get_rounded_int_array(gaussrndm.rvs(3)))
            dists.append(np.linalg.norm(vecs[-1]))

        dists_sort = sorted(dists)
        vecs2 = list()
        for i in range(len(dists_sort)):
            idx = dists.index(dists_sort[i])
            dists[idx] = -99
            vecs2.append(vecs[idx])


        print(self.k)
        print(len(vecs2))
        #tmp_dict = sort_dict(tmp_dict, sort_by='keys')
        #print(tmp_dict)
        #mus = list(tmp_dict.values())
        #print(len(mus))
        #quit(-1104)

        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(vecs2[l])
            cls[l].append(list())

        #cls = self.update_means(cls, data)
        tot = 0
        #for c in cls:
        #    print(cls[c])
        #    print(c)
        #    tot += len(cls[c][1])

        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        while change:
            change = False
            change_cnt = 0
            # for each sample pixel
            # find its nearest mean and put in its
            # cluster and adjust that cluster
            # mean toward the new point
            for sample in range(len(data)):
                dis = 999999
                best = None
                for i in cls:
                    if self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if self.epochs < 5:
                            print('')
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        elif np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean().values
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        #cdis = mahalanobis_distance(data[sample], cls[i][0])

                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in this cluster
                # keep going
                if sample in cls[best][1]:
                    continue
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                            break
                    cls[best][1].append(sample)
                    # now update the center
                    cls = self.kohonen_update_cls(cls, data[sample], best, epsln=.001)
            # once we are done with this run adjust means
            # cls = self.update_means(cls, data)
            # at end of loop see what % of points changed
            # if less than threshold stop
            if self.epochs > 0 and self.epochs%10 == 0:
                #epsln = epsln *.1
                epsln = self.epsilon(.001, .0001, self.epochs, 40)
                print('-----------------------------epsilon', epsln)
            if (change_cnt/self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt/self.size))
            elif self.epochs%50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt/self.size), epsln))
            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def init_random_sample(self, ):
        print('rndm samp')
        print(self.df)
        cp = self.df.copy().values
        np.random.shuffle(cp)
        print(cp)
        new_clusters = cp[0:self.k]
        rl = list()
        # create the list of cluster objects
        for c in range(len(new_clusters)):
            rl.append(cluster(self.k, rc=c, value=new_clusters[c]))
        # calculate the cluster distances
        rl = self.calculate_cluster_diffs(rl)
        return rl
    def normal(self, ):
        pass
    def check_init(self, data):
        if self.init is 'random' and self.method == 'kmeans':
            # generate k random 1X3 list
            # that are from 0 -256
            print('kmeans')
            return self.kmean_init_run(data)
        elif self.init is 'random' and self.method == 'wta':
            print('wta')
            # generate k random 1X3 list
            # that are from 0 -256
            return self.wta_init_run(data)
        elif self.method == 'kohonen':
            print('kohonen')
            return self.kohonen_init_run(data)
        elif self.init is 'normal':
            pass

    def algo_init(self, data, verbose=True):
        """Will initialize the clusters to just
          start as the different samples
        """
        #rl = list([[0]*self.dimen]*self.size)
        rl = list()
        # create a cluster for every row
        # to start
        for row in range(len(data)):
            rl.append(cluster(k=self.k, rc=row, value=data[row]))
            rl[-1].inhabitants.append(data[row])
        return rl

    def adjust_pic(self, df, cls):
        for cl in cls:
            df[cl] = df[cl].mean(axis=0)
        return df

    def find_my_cluster(self, pt, cls):
        for cl in range(len(cls)):
            if pt in cls[cl]:
                return cl
        return None

    def algo_init2(self, data, verbose=True):
        rl = list()
        print('initializing for algorithmic cluster')
        for row1 in range(len(data)-1):
            #if row1 > 20:
            #    break
            for row2 in range(row1+1, len(data)):
                if row1 != row2:
                    rl.append([ np.linalg.norm(data[row1] - data[row2]), int(row1), int(row2)])

        #dummy = np.array(rl[0:10])
        #if verbose:
        #    print('the dummy is ')
        #    print(dummy)
        #dummy2 = col_sort(dummy, 0)
        #if verbose:
        #    print('the dummy2 is ')
        #    print(dummy2)

        rl = col_sort(np.array(rl))
        print('it')
        sound_alert_file(r'')
        print(rl[0:5])
        cls = list([[]]*self.size)

        for idx in range(self.size):
            cls[idx].append(idx)
        for edge in rl:
            cl1 = self.find_my_cluster(edge[1], cls)
            cl2 = self.find_my_cluster(edge[2], cls)
            if cl1 == cl2:
                #already in the same group
                continue
            else:
                cls.append(cls[cl1] + cls[cl2])
                del cls[cl1]
                del cls[cl1]
            if len(cls) == self.k:
                break

        return self.adjust_pic(data, cls)

    def check_method(self, data, verbose=False):
        if self.method is 'algo':
            if verbose:
                print('checked, algo')
            return self.algo_init2(data)
        else:
            return self.check_init(data)
    def rescale_ppm(self, df, cls):
        new_image = None
        # for all of my clusters
        # go through the pixels that belong to it
        # and change thier color values to the clusters
        # color values
        for c in cls:
            val = cls[c][0]
            to_fix = cls[c][1]
            df[to_fix] = val
        #for clstr in self.my_clusters:
        #    for pix in clstr.inhabitants:
        #        self.df[pix] = clstr.value
        return df
class cluster():
    """Represents an individual cluster"""
    def __init__(self, k, rc, value):
        self.k = k                      # number of sibling cluster
        self.value = value              # the current value of the mean I hold
        self.row=rc                     # the row in the toplogical grid I'm in.
        self.cluster_dist = dict()      # distances to the other clusters
        self.inhabitants = list()       # the row number of samples in this cluster
    def get_size(self):
        return len(self.inhabitants)
    def k_mean_calculate_mean(self, df):
        self.value = df[self.inhabitants, ].mean(axis=0)
    def wta_calculate_mean(self, df, eta):
        pass
    def kohonen_calculate_mean(self, df, eta, phi, pho_std):
        pass

class cluster_algos():
    """My collection of clustering algorithms"""
    def __init__(self, df, method='algo', init='random', k=None, distance_type='city_block', distance_metric='dmean'):
        self.df = df                                    # the data we will be working with
        self.my_clusters = clusters(k=k, df=self.df, method=method, init=init, distance_type=distance_type, distance_metric=distance_metric)      # my collection of clusters
        self.k = k                                      # desired number of clusters
    def algorithmic_cluster(self, ):
        """Algorithmic clustering"""

        # while the number of clusters is < k
        cnt = 0
        # do my algorithmic thing yo !!!
        # i.e. run throu some number of epochs or until the desired
        # number of k's is reached
        while self.my_clusters.check_size() > self.k:
            if cnt%(1000) == 0: # shows every thousandth cluster
                print('There are {} clusters'.format(self.my_clusters.check_size()))
            cnt += 1
            # tell the clusters to perform and epoch
            # this will conjoin the closest groups
            # two at a time
            self.my_clusters.perform_epoch()
            # TODO: create a conversion method to
            #  convert old image into rescaled one
            self.my_clusters.rescale_ppm(self.df)

    def finish_init(self):
        pass
    def fit(self, cmodel):
        pass
    def predict(self, X):
        pass
    def score(self, X, Y):
        pass

        # takes the known or Training set and the testing set
        def calculate_distances(self, cluster_means, samples, dist_dic):
            distances_dict = {}

            # iterate through samples of test set
            # calculating the distances between each
            # sample and all other samples in the training set
            for sample1 in range(len(samples)):
                # print(df.iloc[sample1, :])
                # print('==================================')
                # print('==================================')
                # print('==================================')

                # create a dictionary for this sample this will store the distances
                distances_dict[sample1] = {}
                for sample2 in range(len(cluster_means)):
                    # calculate the distance and store it in the dictionary for this entry
                    # print(df.iloc[sample2, :])
                    # print(np.linalg.norm(df_te.iloc[sample1, :].values - df_tr.iloc[sample2, :].values))
                    # print(self.euclidian_dist(df_tr.iloc[sample2, :].values, df_te.iloc[sample1, :].values))
                    # distances_dict[sample1][sample2] = self.euclidian_dist(df_tr.iloc[sample2, :].values, df_te.iloc[sample1, :].values)
                    distances_dict[sample1][sample2] = np.linalg.norm(
                        samples.iloc[sample1, :].values - cluster_means.iloc[sample2, :].values)

                # distances_dict[sample1] = sorted(distances_dict[sample1].items(), key=lambda kv: kv[1])
                distances_dict[sample1] = dict(sorted(distances_dict[sample1].items(), key=operator.itemgetter(1)))
                # distances_dict[sample1] = sorted(distances_dict[sample1].items(), key=operator.itemgetter(1))
                # print(distances_dict[sample1])
            return distances_dict

class Gknn():
    def __init__(self, k=10, dist_metric='euclidean'):
        self.k=k
        self.X=None
        self.y=None
        self.cov=None
        self.inv_cov=None
        self.dist_metric=dist_metric
    def fit(self, X, y):
        self.y = y
        self.X = X
        self.cov = X.cov()
        self.inv_cov = np.linalg.inv(self.cov)

    def predict(self, X):
        dist_dic = self.calculate_distances(X)
        real = self.y.values.flatten().tolist()
        candidates = list(set(real))
        final_tallys = list()
        projected = list()
        yp = list()
        for zone in dist_dic:
            nn = list(dist_dic[zone].keys())
            votes = self.y.values[nn, :].flatten().tolist()
            ballot = {}
            for c in candidates:
                ballot[c] = votes.count(c)
            yp.append(sort_dict(ballot) )

    def calculate_distances(self, X):
        cov=None
        if type_check(X, against='dataframe'):
            cov = self.X.cov()
        else:
            cov = pd.DataFrame(X).cov()

        ret_dict = {}
        for i in range(len(X)):
            cdl = {}
            for j in range(len(self.X)):
                #cdl[j] = mahalanobis_distance(X[i], self.X.values[j], )
                #cdl[j] = np.linalg.norm(X[i]-self.X.values[j])
                cdl[j] = self.get_distance(X[i], j)
                cdl = sort_dict(cdl)
                cnt = 0
                ndl = {}
                for ky in cdl:
                    ndl[ky] = cdl[ky]
                    cnt += 0
                if cnt == self.k:
                    break
            ret_dict[i] = ndl
        return ret_dict

    def get_distance(self, x, j):
        if self.dist_metric == 'city_block':
            return np.linalg.norm(x - self.X.values[j])
        elif self.dist_metric == 'mahalanobis':
            return mahalanobis_distance(x, self.X.values[j], self.inv_cov)
        else:
            return euclidean_distance(x, self.X.values[j], np.mean(self.X.values.std(axis=0)))






class G_NN():
    def __init(self):
        self.data=None
        self.X = None
        self.y = None
        self.Trainset= None
        self.Testset=None


class RF_REG_CLF:
    default_RFR_params = {
        'n_estimators': 10,
        'criterion': 'mae',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 10,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0.0,
        'max_samples': None
    }
    score_ops = ['clf', 'other', 'reg']
    def __init__(self, params=None, learner_type='reg', rate_var='hu_own'):
        if params is None:
            params = self.default_RFR_params
        else:
            pc = self.default_RFR_params.copy()
            for p in params:
                pc[p] = params[p]
            params = pc

        self.RF_reg = build_RFC_RFR(params=params, learner_type=learner_type)
        self.params=params
        self.yreg_p = None
        self.pred_clf = None
        self.acc = 0
        self.sen = 0
        self.spe = 0
        self.cm = 0
        self.mae = 0.0
        self.mse = 0.0
        self.err = 0
        self.fitted = False

    def fit(self, X, Y, store_error=False, rate_col='hu_own'):
        self.RF_reg.fit(X, Y)
        self.Xtr=X
        self.Ytr=Y
        if store_error:
            mse = mean_squared_error(Y, self.predict(X, mode='other',
                                                          rate_col=rate_col))
            self.err = mse
        else:
            self.err = 0
        self.fitted = True
        return
    def predict(self, X, mode='other', N=1, rate_col='hu_own', use_err=True, penalty=1, use_diff=False):
        if not self.fit:
            print('The learner has not been fit, ending program')
            quit(-11)
        if mode == 'clf':
            yreg_p = self.RF_reg.predict(X)
            self.yreg_p = pd.DataFrame({'yreg':yreg_p})
            # generate an all zero array of the correct shape/length
            # then if the predicted regression value >= N/rate_col
            # then predict 1, else predict 0
            pred_clf = np.zeros(len(self.yreg_p))
            print('performing relationship prediction')
            if use_err:
                pred_clf[abs(self.yreg_p['yreg'].values - self.err*penalty)  >= N/X[rate_col]] = 1
            if use_diff:
                pred_clf[abs(self.yreg_p['yreg'].values - N / X[rate_col]) >= self.err*penalty] = 1
            else:
                pred_clf[abs(self.yreg_p['yreg'].values ) >= N / X[rate_col]] = 1
            self.pred_clf = pred_clf
            print('print the first 5: {}'.format(pred_clf[0:5]))
            return pred_clf
        else:
            return self.RF_reg.predict(X)
    def score(self, X, ytrue, mode='other', N=1, rate_col='hu_own',  use_err=True,
              penalty=1, use_diff=False):
        ypred = self.predict(X, mode=mode, N=N, rate_col=rate_col)
        if mode == 'clf':
            self.acc = accuracy_score(ytrue, ypred)
            self.cm = confusion_matrix(ytrue, ypred)
            self.sen = self.cm[1][1]/self.cm[1].sum()
            self.spe = self.cm[0][0] / self.cm[0].sum()
            self.evs = explained_variance_score(ytrue, ypred)
            return self.acc
        else:
            self.mae = mean_absolute_error(ytrue, ypred)
            self.mse = mean_squared_error(ytrue, ypred)
            self.evs = explained_variance_score(ytrue, ypred)
            return self.mse
    def visualize_classification_performance(self, X, y, labels=None, normalize='true',
                                             sample_weight=None, display_labels=None, values_format=None,
                                             include_values=True, xticks_rotation='horizontal',
                                             cmap='summer', ax=None):
        if self.cm == 0:
            print('confusion matrix not created run score or predict in clf mode')
            return
        metrics.plot_confusion_matrix(self.RF_reg, X, y, labels=labels, sample_weight=sample_weight, normalize=normalize,
                                      display_labels=display_labels, include_values=include_values,
                                      xticks_rotation=xticks_rotation, values_format=values_format,
                                      cmap=cmap, ax=ax)


def build_RFC_RFR(params=None, learner_type='clf'):
    default_RFR_params = {
        'n_estimators': 10,
        'criterion': 'mae',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 10,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0.0,
        'max_samples': None
    }
    default_RF_params = {
        'n_estimators': 100,
        'criterion': 'entropy',  # criterion{“gini”, “entropy”}, default=”gini”
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': 10,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'class_weight': None,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

    # TODO: sklearn page: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if params is None:
        if learner_type == 'clf':
            params = default_RF_params
        else:
            params = default_RFR_params

    if params is not None:
        if learner_type == 'clf':
            for p in params:
                if p in params:
                    default_RF_params[p] = params[p]
            params = default_RF_params
        else:
            for p in params:
                if p in params:
                    default_RFR_params[p] = params[p]
            params = default_RFR_params

    if learner_type in ['clf', 'classifier']:
        rfc = RandomForestClassifier(
                             n_estimators = params['n_estimators'],
                             criterion = params['criterion'],
                             max_depth = params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf = params['min_samples_leaf'],
                             min_weight_fraction_leaf = params['min_weight_fraction_leaf'],
                             max_features=params['max_features'],
                             max_leaf_nodes=params['max_leaf_nodes'],
                             min_impurity_decrease=params['min_impurity_decrease'],
                             min_impurity_split=params['min_impurity_split'],
                             bootstrap=params['bootstrap'],
                             oob_score=params['oob_score'],
                             n_jobs=params['n_jobs'],
                             random_state=params['random_state'],
                             verbose=params['verbose'],
                             warm_start=params['warm_start'],
                             class_weight=params['class_weight'],
                             )
    else:
        params = default_RFR_params
        rfc = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            min_impurity_decrease=params['min_impurity_decrease'],
            min_impurity_split=params['min_impurity_split'],
            bootstrap=params['bootstrap'],
            oob_score=params['oob_score'],
            n_jobs=params['n_jobs'],
            random_state=params['random_state'],
            verbose=params['verbose'],
            warm_start=params['warm_start'],
        )
    return rfc

# my version of a solar forrest (classifier + regressor for regression)
class convergent_forest:
    def __init__(self, paramsC, paramsR):
        self.RFC = build_RFC_RFR(paramsC, learner_type='clf')
        self.RFR = build_RFC_RFR(paramsR, learner_type='LR')
        self.regression_feat_imp=None
        self.classification_feat_imp=None
        self.accuracy=list()
        self.sensitivity=list()
        self.specificity=list()
        self.variance_exp_C = list()
        self.mae=list()
        self.mse=list()
        self.variance_exp_R = list()
        self.X, self.yc, self.yr = None, None, None
        self.XtrC, self.XtsC, self.ytsC, self.ytrC, self.ytsR, self.ytrR = None, None, None, None, None, None
        self.ypredC = list()
        self.ypredR = list()
        self.cm = None
        return

    def remove_target(self, feats, targetC, verbose=False):
        if targetC in feats:
            del feats[feats.index(targetC)]
        return feats


    def generate_forest_data(self, df, feats, X=None, y=None,  targetR='PV_HuOwn',
                             targetC='Adoption', verbose=True, tr=(.75, ) ):
        # remove classification target
        self.featsC = self.remove_target(feats, targetC, verbose=verbose)
        feats = self.featsC
        if X is not None and y is not None:
            self.X, self.yc = X, y
        else:
            self.X = df.filter(items=self.featsC)
            self.yc =df.filter(items=[targetC])

        tr = tr[0]
        ts = np.around(1 - tr, 2)
        self.XtrC, self.XtsC, self.ytrC, self.ytsC = train_test_split(self.X, self.yc, test_size=ts,
                                              train_size=tr, stratify=self.yc)

        # Pull out your predictors from your targets for the training set

        self.XtrC = pd.DataFrame(self.XtrC, columns=feats)
        self.ytrC = pd.DataFrame(np.array(self.ytrC).flatten(), columns=[targetC])

        # Pull out your predictors from your targets for the testing set
        self.XtsC = pd.DataFrame(self.XtsC, columns=feats)
        self.ytsC = pd.DataFrame(np.array(self.ytsC).flatten(), columns=[targetC])

        # now make regression data
        # remove regression target
        feats = self.remove_target(feats, targetR,)
        self.ytrR = self.XtrC.filter(items=[targetR])
        self.ytsR = self.XtsC.filter(items=[targetR])
        self.XtrC = self.XtrC.filter(items=feats)
        self.XtsC = self.XtsC.filter(items=feats)


        return self.XtrC, self.XtsC, self.ytrC, self.ytsC, self.ytrR, self.ytsR


    def fit(self, X, yR, yC):
        self.RFC.fit(X, yC)
        self.RFR.fit(X, yR)
        return

    def predict(self, X, ):
        yp = list()
        self.ypredC, self.ypredR = list(), list()
        for ix in range(len(X)):
            # predict with RFC
            self.ypredC.append(self.RFC.predict(X.values[ix:ix+1, :]))

            if self.ypredC[-1] == 0:
                yp.append(0)
            else:
                yp.append(self.RFR.predict(X.values[ix:ix+1, :]))
        self.ypredR = yp
        return yp

    def score(self, X, yr, yc, score=['accuracy', ], verbose=True):
        yp = self.predict(X)
        if 'sensitivity' in score or 'specificity' in score:
            cm = confusion_matrix(yc, self.ypredC)
            self.cm = cm
            spec = cm[0][0]/cm[0].sum()
            sen = cm[1][1] / cm[1].sum()
        scores = {}
        for scr in score:
            if scr == 'accuracy':
                scores[scr] = accuracy_score(yc, self.ypredC)
            if scr == 'sensitivity':
                scores[scr] = sen
            if scr == 'specificity':
                scores[scr] = spec
            if scr == 'mae':
                scores[scr] = mean_absolute_error(yr, self.ypredR)
            if scr == 'mse':
                scores[scr] = mean_squared_error(yr, self.ypredR)
            if scr == 'r2':
                scores[scr] = explained_variance_score(yr, self.ypredR)
        if len(scores) == 1:
            return scores[score[0]]
        return scores



class naive_bayes_classifier():
    def __init__(self, classifiers=(), verbose=True):
        self.cms = list()
        self.LUT = None
        self.clsf = [clf for clf in classifiers]
        self.accuracies = list()
        self.sensitivities = list()
        self.specificities = list()
        self.evs = list()
        self.r2 = list()
    # will fit all classifiers with given training
    # set and generate confusion matrices for each
    def fit(self, X, y, verbose=False):
        if isinstance(X, type(list())):
            cnt = 1
            for clf, xt in zip(range(len(self.clsf)), X):
                print('fitting learner {}'.format(cnt))
                cnt += 1
                self.clsf[clf].fit(xt, y)
                yp = self.clsf[clf].predict(xt)
                cm = confusion_matrix(y, yp)
                negc = cm[0].sum()
                posc = cm[1].sum()
                self.cms.append(cm)
                spec = cm[0][0]/negc
                sen = cm[1][1]/posc

                self.accuracies.append(accuracy_score(y, yp))
                self.sensitivities.append(sen)
                self.specificities.append(spec)
                self.evs.append(explained_variance_score(y, yp))
        else:
            for clf in range(len(self.clsf)):
                self.clsf[clf].fit(X,y)
                yp = self.clsf[clf].predict(X)
                cm = confusion_matrix(y, yp)
                self.cms.append(cm)
                negc = cm[0].sum()
                posc = cm[1].sum()

                spec = cm[0][0] / negc
                sen = cm[1][1] / posc

                self.accuracies.append(accuracy_score(y, yp))
                self.sensitivities.append(sen)
                self.specificities.append(spec)
                self.evs.append(explained_variance_score(y, yp))
        #self.cms = self.generate_cms(X, y)
        self.LUT = self.naive_bayes_cm_fnc(self.cms)
        return

    def generate_cms(self, X, y, verbose=False):
        """
            will generate the confusion matrices for the baysian
            classifier
        :param X: predictor
        :param y: target
        :param verbose:
        :return: list of confusion matricies, entry  i is classifier i's
        """
        return [confusion_matrix(y, clsf.predict(X)) for clsf in self.clsf]

    def set_cms(self, cms):
        for clf in range(len(self.clsf)):
            self.clsf[clf].cm = cms[clf]

    def predict(self, x):
        """ Will go through classifiers, getting prediction lists
        :param x:
        :return:
        """
        # will contain a list of lists where list i is for
        # observation i, and the contents of list i are
        # the predictions for observation i from the classifiers
        predictions = list()

        if type(x) == type(pd.DataFrame([])):
            print('fix the object')
            x = x.values

        for obs in range(len(x)):
            # create list i
            predictions.append(list())
            # move through classifiers generating predictions for
            # observation i
            for clsf in self.clsf:
                predictions[obs].append(clsf.predict([x[obs]]))
                #print(predictions[obs])
        tl = []
        # now go through the predictions for each observation using the
        # look up table to make the final prediction
        yp = list()
        for obs in range(len(predictions)):
            #print(self.LUT[predictions[obs][0], predictions[obs][1]].tolist())
            yp.append( self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0].index(max(self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0])))

        #print(yp)
        return yp

    def predictM(self, Xts):
        """ Will go through classifiers, getting prediction lists
        :param x:
        :return:
        """
        # will contain a list of lists where list i is for
        # observation i, and the contents of list i are
        # the predictions for observation i from the classifiers
        predictions = list()

        for obs in range(len(Xts[0])):
            # create list i
            predictions.append(list())
            # move through classifiers generating predictions for
            # observation i
            for clsf, cX in zip(self.clsf, Xts):
                predictions[obs].append(clsf.predict([cX.values[obs]]))
                #print(predictions[obs])
        tl = []
        # now go through the predictions for each observation using the
        # look up table to make the final prediction
        yp = list()
        for obs in range(len(predictions)):
            #print(self.LUT[predictions[obs][0], predictions[obs][1]].tolist())
            yp.append( self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0].index(max(self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0])))

        #print(yp)
        return yp

    def naive_bayes_cm_fnc(self, conmats, verbose=True):
        # print(cm1.transpose().reshape((9,)))
        # print(cm1[0:, 0])
        # print(cm2[0:, 0])
        # print(cm1[0:, 0]*cm2[0:,0])

        nbcml = []                      # naive bayes confusion matrix list
        for cms in conmats:
            nbcml.append(self.nb_cm(cms).prob_table)
        nb1 = self.nb_cm(conmats[0])           # confusion matrix object 1
        nb2 = self.nb_cm(conmats[1])           # confusion matrix object 2

        if False:
            print('cm1')
            print(nb1.cm)
            print('cm2')
            print(nb2.cm)
            print('prob table 1')
            print(nb1.prob_table)
            print('prob table 2')
            print(nb2.prob_table)
            print('cm1:')
            print(cm1)
            print('cm2:')
            print(cm2)
            print('product')
            print(cm1 * cm2)

#        shp = cm1.shape[0]
#        shp = nbcml[0].cm.shape[1]
        shp = conmats[0].shape[0]
        # create empty look up table
        tupparam = tuple([shp for i in range(shp+1)])
        look_up = np.empty(tupparam)
        print(look_up)

        # print(nb_mat)
        # nb_mat[0,0] = cm1[:,0]*cm2[:,0]
        # print(nb_mat[0,0,:])
        # print(nb_mat[0,1,:])

        '''
        for col1 in range(shp):
            for colb in range(shp):
                look_up[col1, colb] = nb1.prob_table[:, col1] * nb2.prob_table[:, colb]
        '''

        for col1 in range(shp):
            for colb in range(shp):
                #look_up[col1, colb] = nb1.prob_table[:, col1] * nb2.prob_table[:, colb]
                look_up[col1, colb] = nbcml[0][:, col1]
                for nb1 in nbcml[1:]:
                    look_up[col1, colb] *= nb1[:, colb]
        return look_up

    class nb_cm():
        """represents a confusion matrix from a classifier"""
        def __init__(self, cm):
            self.cm = cm                                    # the confusion matrix stored
            self.row_sums = self.calc_row_sums()            # the row sums (class counts) for each class
            self.prob_table = self.create_prob_table()      # probability table used to make look up table
        # counts the number of each class in the confusion matrix
        def calc_row_sums(self):
            return [sum(r) for r in self.cm]
        # probability table created from confusion matrix
        def create_prob_table(self):
            return self.cm / self.row_sums

class council_of_trees:
    def __init__(self, trees, ):
        self.cms = list()

        self.clsf = trees
        self.accuracies = list()
        self.sensitivities = list()
        self.specificities = list()
        self.evs = list()
        self.r2 = list()

    # will fit all classifiers with given training
    # set and generate confusion matrices for each
    def fit(self, X, y, Xts=None, yts=None):
        cnt = 0
        for clf, xt in zip(range(len(self.clsf)), X):
            print('fitting learner {}'.format(cnt))

            self.clsf[clf].fit(xt, y)
            if Xts is not None:
                yp = self.clsf[clf].predict(Xts[clf])
                cm = confusion_matrix(yts, yp)
                self.accuracies.append(accuracy_score(yts, yp))
                self.evs.append(explained_variance_score(yts, yp))
            else:
                yp = self.clsf[clf].predict(xt)
                cm = confusion_matrix(y, yp)
                self.accuracies.append(accuracy_score(y, yp))
                self.evs.append(explained_variance_score(y, yp))
            print(cm)
            print('----------------------\n')
            negc = cm[0].sum()
            posc = cm[1].sum()
            self.cms.append(cm)
            spec = cm[0][0] / negc
            sen = cm[1][1] / posc
            self.sensitivities.append(sen)
            self.specificities.append(spec)
            cnt += 1
        return

    def generate_cms(self, X, y, verbose=False):
        """
            will generate the confusion matrices for the baysian
            classifier
        :param X: predictor
        :param y: target
        :param verbose:
        :return: list of confusion matricies, entry  i is classifier i's
        """
        return [confusion_matrix(y, clsf.predict(X)) for clsf in self.clsf]

    def set_cms(self, cms):
        for clf in range(len(self.clsf)):
            self.clsf[clf].cm = cms[clf]

    def predict(self, x):
        """ Will go through classifiers, getting prediction lists
        :param x:
        :return:
        """
        # will contain a list of lists where list i is for
        # observation i, and the contents of list i are
        # the predictions for observation i from the classifiers
        predictions = list()

        for obs in range(len(x[0])):
            # create list i
            predictions.append(list())
            # move through classifiers generating predictions for
            # observation i
            for clsf, cX in zip(self.clsf, x):
                predictions[obs].append(clsf.predict([cX.values[obs]]))
        print(len(predictions))
        voted_p = list()
        for pcl in range(len(predictions)):
            if predictions[pcl].count(0) > predictions[pcl].count(1):
                voted_p.append(0)
            else:
                voted_p.append(1)

        return voted_p

    def predictM(self, Xts):
        """ Will go through classifiers, getting prediction lists
        :param x:
        :return:
        """
        # will contain a list of lists where list i is for
        # observation i, and the contents of list i are
        # the predictions for observation i from the classifiers
        predictions = list()

        for obs in range(len(Xts[0])):
            # create list i
            predictions.append(list())
            # move through classifiers generating predictions for
            # observation i
            for clsf, cX in zip(self.clsf, Xts):
                predictions[obs].append(clsf.predict([cX.values[obs]]))
                # print(predictions[obs])
        tl = []
        # now go through the predictions for each observation using the
        # look up table to make the final prediction
        yp = list()
        for obs in range(len(predictions)):
            # print(self.LUT[predictions[obs][0], predictions[obs][1]].tolist())
            yp.append(self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0].index(
                max(self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0])))

        # print(yp)
        return yp

    def naive_bayes_cm_fnc(self, conmats, verbose=True):
        # print(cm1.transpose().reshape((9,)))
        # print(cm1[0:, 0])
        # print(cm2[0:, 0])
        # print(cm1[0:, 0]*cm2[0:,0])

        nbcml = []  # naive bayes confusion matrix list
        for cms in conmats:
            nbcml.append(self.nb_cm(cms))
        nb1 = self.nb_cm(conmats[0])  # confusion matrix object 1
        nb2 = self.nb_cm(conmats[1])  # confusion matrix object 2

        if False:
            print('cm1')
            print(nb1.cm)
            print('cm2')
            print(nb2.cm)
            print('prob table 1')
            print(nb1.prob_table)
            print('prob table 2')
            print(nb2.prob_table)
            print('cm1:')
            print(cm1)
            print('cm2:')
            print(cm2)
            print('product')
            print(cm1 * cm2)

        #        shp = cm1.shape[0]
        #        shp = nbcml[0].cm.shape[1]
        shp = conmats[0].shape[0]
        # create empty look up table
        tupparam = tuple([shp for i in range(shp + 1)])
        look_up = np.empty(tupparam)
        print(look_up)

        # print(nb_mat)
        # nb_mat[0,0] = cm1[:,0]*cm2[:,0]
        # print(nb_mat[0,0,:])
        # print(nb_mat[0,1,:])

        '''
        for col1 in range(shp):
            for colb in range(shp):
                look_up[col1, colb] = nb1.prob_table[:, col1] * nb2.prob_table[:, colb]
        '''

        for col1 in range(shp):
            for colb in range(shp):
                # look_up[col1, colb] = nb1.prob_table[:, col1] * nb2.prob_table[:, colb]
                look_up[col1, colb] = nbcml[0][:, col1]
                for nb1 in nbcml[1:]:
                    look_up[col1, colb] *= nb1.prob_table[:, colb]
        return look_up

    class nb_cm():
        """represents a confusion matrix from a classifier"""

        def __init__(self, cm):
            self.cm = cm  # the confusion matrix stored
            self.row_sums = self.calc_row_sums()  # the row sums (class counts) for each class
            self.prob_table = self.create_prob_table()  # probability table used to make look up table

        # counts the number of each class in the confusion matrix
        def calc_row_sums(self):
            return [sum(r) for r in self.cm]

        # probability table created from confusion matrix
        def create_prob_table(self):
            return self.cm / self.row_sums


class GLIN_Regressor(Learner):
    def __init__(self, X, Y, Xts, Yts, w=None, c = 0, intercept=True, eta=.0005, etamin=.000001,
                 etamax=.0001, kmax=200, eta_dec=True, epochs=90000, cost_func='mse', wgt='zero', tol=1e-3, lm=4):
        super().__init__()
        self.data = None
        self.intercept = intercept
        self.w = w
        self.wgt = wgt
        self.c = c
        self.X = X
        self.Y = Y
        self.Xts = Xts
        self.Yts = Yts
        self.tol = tol
        self.Ymean = Y.mean(axis=0)
        self.test_epochs = None
        self.test_mae = None
        self.test_cod = None
        self.test_rmse = None
        #print(self.Ymean)
        self.Ymeants = Yts.mean(axis=0)
        #print(self.Ymeants)
        self.best_MAE = 999999
        self.N = len(X)
        print('N:', self.N)
        self.d = X.shape[1]
        self.eta = eta
        self.epochs = epochs
        self.etamax=etamax
        self.etamin=etamin
        self.kmax=kmax
        self.eta_dec=eta_dec
        self.cost_fnc = cost_func
        self.p_scores = None
        self.Vif_scores = None
        self.Rsqr = None
        self.wald_chi = None
        self.best_w = None
        self.best_b = None
        self.best_MSE = 200000000000
        self.best_RMSE = 200000000000
        self.best_COD = 200000000000
        self.best_MAE = 200000000000
        self.best_Rsqr = 20000000000
        self.epoch_stop = -99
        self.lm = lm
        self.finish_init()

    # performs multiple linear regression on the x and y data
    # and returns the generated parameter vector W
    def multi_linear_regressor(self, x_data, y_data):
        x = np.array(x_data, dtype=np.float)
        y = np.array(y_data, dtype=np.float)
        x_transpose = np.transpose(x)
        xtx = np.dot(x_transpose, x)
        xtx_inv = np.linalg.inv(xtx)
        xtx_inv_xt = np.dot(xtx_inv, x_transpose)
        w = np.dot(xtx_inv_xt, y)
        return w

    def wgt_predict(self, wgt, X):
        return np.dot(X.transpose(), wgt)

    def sigmoid(self, X):
        return 1/(1-np.e**(-X))

    def finish_init(self):
        """
            Sets up the weights and b
        :return:
        """
        if self.w is None:
            if self.wgt == 'random':
                self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
            elif self.wgt== 'zero':
                self.w = np.array([0] * self.d)
            elif self.wgt == 'ols':
                self.w = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
                self.w = np.dot(self.w, self.Y)
            """
            if self.intercept:
                if self.wgt == 'random':
                    # get a normally distributed randomized weight vector
                    self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
                    self.w = np.array(self.w + [1])
                else:
                    #b = list([[1]]*(self.N))                    # add intercept
                    #self.X = np.hstack((self.X, b))
                    #self.d = self.d + 1
                    #b = list([[1]]*(len(self.Xts)))                    # add intercept
                    #self.Xts = np.hstack((self.Xts, b))
            else:
                self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
                self.w = np.array(self.w)
        self.wd = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
        self.wd = np.dot(self.wd, self.Y)
        print('wd shape', self.wd.shape)
        print(self.wd)
        """

    def predicted_weights(self, yp):
        wd_a = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
        return np.dot(wd_a, yp)

    def predict_score(self,X, ytr, ypr, verbose=False):
        pass

    def report_scores(self, ytr, ypr, verbose=False, Ymean=None):
        if Ymean is None:
            Ymean = self.Ymean
        n = len(ytr)
        mmse = self.MSE(ytr, ypr, n)
        print('n',n)
        print('ypr', len(ypr))
        print('          ----------------RMSE:',np.sqrt(mmse))
        print('          -----------------MSE: ', mmse)
        print('          ---------sklearn MSE:', metrics.mean_squared_error(ytr, ypr))
        print('          -------var explained:', metrics.explained_variance_score(ytr, ypr))
        print('          -----------------MAE:', self.MAE(ytr, ypr, n))
        print('          -----------------CD:', self.R2(ytr, ypr, Ymean))
        print('          ---------sklearn r2:', metrics.r2_score(ytr, ypr))
        print('          -----------------R^2', self.Rvar(ytr, ypr, Ymean))
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------\n')

    def cost_derivative(self, X, ytruth, ypred, cost_fnc='mae'):
        if cost_fnc == 'mae':
            maePrime_w = -1/len(ytruth) * np.dot((ytruth-ypred)/(abs(ytruth - ypred)),X)
            maePrime_b = -1 / len(ytruth) * sum([(yt-yp)/abs(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [maePrime_w, maePrime_b]
        if cost_fnc == 'mse':
            print('mse')
            msePrime_w = -2/len(ytruth) * np.dot((ytruth-ypred), X)
            msePrime_b = -2 / len(ytruth) * sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [msePrime_w, msePrime_b]


    def fit(self,solver='mae'):
        #est = .0001
        #est = .00992
        est = self.eta
        etamax = self.eta
        etamin = self.etamin
        epochs = self.epochs
        kmax = self.kmax
        self.test_cod, self.test_epochs, self.test_mae, self.test_rmse = list(), list(), list(), list()
        #self.w = np.array([0]*self.d)
        #self.w = self.wd
        #print(self.w)
        n = self.N
        print('-----------------------N', self.N)
        print(self.cost_fnc)
        d = self.d
        mmse_old = 0
        mmae_old = 0
        old_thresh = list()
        for i in range(epochs):
            #np.random.shuffle(self.X.values)
            # get a prediction
            #print('y means', self.Ymean)
            #print('ytest means', self.Ymeants)
            #print('w\n', self.w)
            yp = np.dot(self.X, self.w) + self.c
            #print('yp\n',yp)
            #print('gf\n', self.Y)
            mmse =self.MSE(self.Y, yp, n)
            mmae =self.MAE(self.Y, yp, n)
            rmse = self.RMSE(self.Y, yp)
            self.test_epochs.append(i)
            self.test_rmse.append(rmse)
            self.test_cod.append(self.R2(self.Y, yp))
            self.test_mae.append(mmae)
            if self.cost_fnc == 'mse':
                old_thresh.append(mmse)
            elif self.cost_fnc == 'mae':
                old_thresh.append(mmae)
            if self.cost_fnc == 'mse' and mmse < self.best_MSE:
                print('--------------------------------------------------------------    New Best MSE:', mmse)
                self.best_MSE = mmse
                self.best_MAE = mmae
                self.best_RMSE = rmse
                self.best_COD = self.R2(self.Y, ypred=yp, ymean=self.Ymean)
                self.best_b = self.c
                self.best_w = self.w
                self.best_score = mmse
                self.best_epoch = i
            if self.cost_fnc == 'mae' and mmae < self.best_MAE:
                print('--------------------------------------------------------------    New Best MAE:', mmae)
                self.best_MAE = mmae
                self.best_MSE = mmse
                self.best_RMSE = rmse
                self.best_COD = self.R2(self.Y, ypred=yp, ymean=self.Ymean)
                self.best_b = self.c
                self.best_w = self.w
                self.best_score = mmae
                self.best_epoch = i


            #yp = self.X*self.w + self.c

            #print('pred',yp)
            #D_m = (-2/n) * sum(np.dot(self.X.transpose(), (self.Y - yp)))
            #D_m = (-2/n) * sum(self.X.values.reshape(self.N, 1) * (self.Y - yp))
            #D_m = (-2/n) * sum((self.Y - yp).values.reshape(self.N, 1)*self.X)
            #D_m = (-2/n) * sum((self.X.reshape(1, self.N))*(self.Y - yp))
            #D_m = (-2/n) * sum((self.X.reshape(1, self.N))*(self.Y - yp))
            #D_m = (-2/n) * sum((self.X.transpose())*(self.Y - yp))
            #print('-----------------------------------------')
            #print(self.X.shape)
            #print(self.Y-yp)

            w_b = self.cost_derivative(self.X, self.Y, yp, self.cost_fnc)

            D_m = w_b[0]
            D_c = w_b[1]
            #D_m = (-1/n) * (np.dot(1/abs(self.Y - yp), self.X))
            #D_c = (-1/n)* (1/sum(abs(self.Y-yp)))
            self.w = self.w - self.eta*D_m
            self.c = self.c - self.eta*D_c
            print('eta: {} -----------------RMSE:'.format(self.eta), np.sqrt(mmse))
            print('Epoch: {} -----------------MSE:'.format(i+1), mmse)
            print('          ---------sklearn MSE:'.format(i+1), metrics.mean_squared_error(self.Y, yp))
            print('          ---------var explained:'.format(i+1), metrics.explained_variance_score(self.Y, yp))
            print('          -----------------MAE', self.MAE(self.Y, yp, n))
            print('          -----------------CD:', self.R2(self.Y, yp, self.Ymean))
            print('          ---------sklearn r2:'.format(i+1), metrics.r2_score(self.Y, yp))
            print('          -----------------R^2', self.Rvar(self.Y, yp, self.Ymean))
            print('-------------------------------------------------------------')
            print('-------------------------------------------------------------\n')
            #if abs(mmse - mmse_old) < .00000000001:
            lm = self.lm
            if len(old_thresh) >= lm and   abs(sum(old_thresh[-lm:])/lm - old_thresh[-1]) < self.tol:
                print('-- -- -- -- -- -- -- ****** thresh met {} ******'.format(abs(sum(old_thresh[-lm:])/lm - old_thresh[-1])))
                break
            if self.cost_fnc == 'mae' and abs(mmae - mmae_old) < self.tol *.00001:
                    #print('thresh met {}'.format(abs(mmse - mmse_old)))
                    print(' ****** thresh met {} ******'.format(abs(mmae - mmae_old)))
                    break
            if self.cost_fnc == 'mse' and abs(mmse - mmse_old) < self.tol * .00001:
                    print(' ****** thresh met {} ******'.format(abs(mmse - mmse_old)))
                    break
            mmse_old = mmse
            mmae_old = mmae
            self.eta = epsilon(emax=etamax, emin=etamin, k=i, kmax=kmax)

    def fit2(self, cmodel):
        cnt = 0
        est = 1/100000
        self.eta = est
        etamax = est
        etamin = est*.01
        kmax = 10000000
        threshold = .1
        dif = 1000000
        #ymean = self.Y.values.mean(axis=0)
        ymean = self.Ymean
        self.w = self.wd
        w = self.w
        while .0001 < dif:
            pred = []
            """
            # go through making predictions correcting the error as you go
            #for x, y, w in zip(self.X, self.Y, self.w.transpose()):
            for x, y in zip(self.X, self.Y):
                # make prediction
                    #print('shape of x')
                    #print(self.wd.transpose().shape[0])
                    #print('wd')
                #print(self.w)
                    #print(self.wd.shape)
                g = np.dot(x, self.w.transpose())
                pred.append(g)
                # get the error of the derirvative

                    #wd = np.dot(np.linalg.inv(np.dot(x.transpose(), x)),x.transpose())
                    #print('w')
                    #print(self.w)
                    #cw = np.dot(wd, self.Y)
                #div = -2*np.linalg.norm(self.w - self.wd)
                #print('g')
                    #print(g)
                    #print('                  y')
                    #print(y)
                    #print('error', div)
                #self.w = self.w - eta*div
                #self.wd = self.wd - eta*div
                #print(self.w)
            # once predict done calculate error and if
            if cnt > 10:
                k = 0
            """

            """
            #score
            sum = 0
            rss = 0
            tss = 0
            for g, y in zip(pred, self.Y):
                sum += (g - y)**2
                rss +=  (g - ymean)**2
                tss += (y-ymean)**2
            print('sum', sum)
            scr = (sum/self.N)
            rsqu = (rss/tss)
            print('MSE')
            print(scr)
            print('rsqur')
            print(1- rsqu)
            """
            yp =   self.predict(self.Xts)
            #print(yp[0:5])
            #print(self.Y[0:5])
            mae, mse, rsqu, rvar, mse_prime =   self.score(ypred=yp)
            #mae, mse, rsqu, rvar, mse_prime =   self.score(ypred=pred)
            print('Epoch: {} eta:{}, mae: {}, mse: {}, R2: {}, Rvar: {}, dif {}'.format(cnt, np.around(self.eta,3), mae, mse, rsqu, rvar, dif))
            old = self.wd
            #print('old')
            #print(old)
            #print('w')
            #print(self.w)
            print('prime')
            print(mse_prime[0][0:self.d])
            print('prime')
            print(mse_prime[1])
            self.wd[0:self.d] = self.wd[0:self.d] - self.eta * mse_prime[0]
            self.wd[self.d] = self.wd[self.d] - self.eta * mse_prime[1]

            #print('old')
            #print(old)
            #print('w')
            #print(self.w)
            dif = abs(np.dot(self.w, old))

            print('')
            self.eta = epsilon(emax=etamax, emin=etamin, k=cnt, kmax=kmax)

            cnt += 1

    def g_OLS(self, x, y):
        pass

    def predict(self, X):
        return np.dot(X, self.best_w) + self.best_b

    def SSE(self, ytrue, ypred):
        return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

    def MSE(self, yt, yp, n=None):
        if n is None:
            n = len(yt)
        return self.SSE(yt, yp)/n
    def RMSE(self, yt, yp, n=None):
        if n is None:
            n = len(yt)
        return sqrt(self.SSE(yt, yp)/n)
    def MAE(self, ytrue, ypred, n=None):
        if n is None:
            n = len(ytrue)
        return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n

    def SSREG(self, ypred, ymean):
        return sum([(yp - ymean) ** 2 for yp in ypred])

    def SSRES(self, ytrue, ypred):
        return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

    def R2(self, ytrue, ypred, ymean=None):
        if ymean is None:
            ymean = self.Ymean
        return 1 - (self.SSRES(ytrue, ypred)/self.SSTOT(ytrue, ymean))

    def Rvar(self, ytrue, ypred, ymean):
        ssreg = self.SSREG(ytrue, ymean=ymean)
        ssres = self.SSRES(ytrue=ytrue, ypred=ypred)
        return (self.SSREG(ypred, ymean)/self.N)/(self.SSTOT(ytrue, ymean)/self.N)
        # return self.SSREG(ypred, ymean)/ (ssres + ssreg)

    def SSTOT(self, ytrue, ymean):
        return sum([(yt - ymean) ** 2 for yt in ytrue])  # scatter total (sum of squares)


    def score(self, ypred, ytrue=None, ymean=None, verbose=False):
        """returns a number of scoring metrics for the predictions from a linear regression
        :param ypred: predicted values from a learner
        :param ytrue: the ground truth values
        :param ymean: the average value for the target variable
        :param verbose: how talkative you want the scoring to be
        :return: mae (mean absolute error), mse(mean square error), R2 (coefficient of determination), R2var (proportion of variance explained)
        """
        if ytrue is None:
            ytrue = self.Yts
        if ymean is None:
            ymean = self.Ymeants

        ssres = sum([(yt-yp)**2 for yp, yt in zip(ytrue, ypred)])   # residual sum of squares (error)
        mae = sum([abs(yt-yp) for yp, yt in zip(ytrue, ypred)])/len(self.Xts) # mean absolute error
        sstot = sum([(yt-ymean)**2 for yt in ytrue])           # scatter total (sum of squares)
        ssreg = sum([(yp-ymean)**2 for yp in ypred])           # sum of sqaures(variance from mean of predictions)
        mse_prime = []
        mse = ssres/len(self.Xts)
        mse_prime.append(-2*sum([np.dot(x,(yt-yp)) for yp, yt, x in zip(ytrue, ypred, self.Xts)])/len(self.Xts))
        mse_prime.append(-2*sum([(yt-yp) for yt, yp in zip(ytrue, ypred)])/len(self.Xts))
        R2 = 1 - (ssres)
        R2var = ssreg/max(.01, (ssres + ssreg))

        return mae, mse, R2, R2var, mse_prime

def random_forest_tester(X_tr, y_tr, X_ts, y_ts, verbose=False, param_grid=None, s=0, cv=5, save_feats=False):
    if param_grid is None:
        param_grid = {
            # 'n_estimators': [1500, 1800, 2000],  # how many trees in forest
            'n_estimators': [1000, 2200],  # how many trees in forest
            # 'max_features': [None, 'sqrt', 'log2'],       # maximum number of features to test for split
            'max_features': [None],  # maximum number of features to test for split
            # 'max_features': ['sqrt'],
            # 'criterion': ['gini'],
            'criterion': ['entropy'],  # how best split is decided
            # 'max_depth': [None, 10, 100, 1000, 10000],            #
            # 'max_depth': [None, 10, 100],                      # how large trees can grow
            # 'max_depth': [None, 10, 20],  # how large trees can grow
            'max_depth': [50, None],  # how large trees can grow
            'oob_score': [True],  #
            # 'min_samples_leaf': [1, 3, 5],                             # The minimum number of samples required to be at a leaf node
            'min_samples_leaf': [1],  # The minimum number of samples required to be at a leaf node
            # 'max_leaf_nodes': [None, 2, 10],
            'max_leaf_nodes': [None],
            'min_weight_fraction_leaf': [0],  #
            # 'min_samples_split': [2, .75],
            'min_samples_split': [2],
            'min_impurity_decrease': [0, .01],
            'random_state': [None],
            # 'class_weight': [None,]
            'class_weight': ['balanced_subsample', 'balanced', None, {0: .4, 1: .6}]
        }

    RF_clf0 = RandomForestClassifier()
    scorers0 = {
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
        'precision_score': make_scorer(precision_score),
        'confusion_matrix': make_scorer(confusion_matrix)
    }
    scorersSens = {
        'recall_score': make_scorer(recall_score)
    }
    scorersAcc = {
        'accuracy_score': make_scorer(accuracy_score)  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
    }
    scorersPrec = {
        'precision_score': make_scorer(precision_score),
        # TP/(TP+FP), a metric of models ability to not miss label a positive
    }
    scorers = [scorersAcc, scorersSens, scorersPrec]
    scr = ['accuracy_score', 'recall_score', 'precision_score']
    GSCV_clf0 = GridSearchCV(estimator=RF_clf0, param_grid=param_grid, cv=cv, scoring=scorers[s], refit=scr[s])
    GSCV_clf0.fit(X_tr, y_tr)
    print('Scoring for {:s}'.format(scr[s]))
    print('ZBest Params:')
    print(GSCV_clf0.best_params_)
    print('best score: ',GSCV_clf0.best_score_)
    RF_clfstd = GSCV_clf0.best_estimator_
    feature_impz = RF_clfstd.feature_importances_
    ypz = RF_clfstd.predict(X_ts)
    feates = viz.display_significance(feature_impz, X_tr.columns.values.tolist(), verbose=True)
    if save_feats:
        pd.DataFrame({'variables': list(feates.keys()), 'Sig': list(feates.values())}).to_excel(
            'RandomForest_Feature_significance_{}_.xlsx'.format(today_is()))
    accuracy, scores, posneg, = bi_score(ypz, y_ts, vals=[0, 1], classes='')
    print('Sensitivity:', posneg['sen'])
    viz.show_performance(scores=scores, verbose=True)
    print('=================================================================================================')
    print('=================================================================================================')

def logistic_tester(X_tr, y_tr, X_ts, y_ts, verbose=False, param_grid=None, pg=1, s=0, cv=5, save_feats=False):
    if param_grid is None:
        # set up parameter grid for grid search testing
        param_gridB = {'penalty': ['elasticnet'],
                           'dual': [False],
                           'tol': [1e-4, 1e-6],
                           'Cs': [10],
                           'fit_intercept': [True],
                           'class_weight': ['balanced', {0: .6, 1: .4}, {0: .4, 1: .6}],
                           'solver': ['saga'],
                           'max_iter': [5000, 100000],
                              }
        param_gridA = {'penalty': ['l2'],
                           'dual': [False],
                           'tol': [1e-1, 1e-3],
                           'Cs': [10, 1, 5],
                           'cv': [3, 5],
                           'fit_intercept': [True],
                           'class_weight': ['balanced', {0: .5, 1: .5}, {0: .55, 1: .45}],
                           'solver': ['newton-cg', 'lbfgs', 'sag'],
                           'max_iter': [1000, 5000, 10000],
                              }
        param_gridl = {'penalty': ['l1'],
                           'dual': [False],
                           'tol': [1e-2, 1e-3],
                           'Cs': [10],
                           'cv': [3, 5],
                           'fit_intercept': [True],
                           # 'class_weight': [{0: .5, 1: .5}, {0: .6, 1: .4}],
                           'class_weight': ['balanced', {0: .5, 1: .5}, {0: .6, 1: .4}],
                           'solver': ['liblinear', 'saga'],
                           # 'max_iter': [1000, 2000, 5000],
                           'max_iter': [900, 2000, 5000],
                              }
        param_grid = [param_gridB,param_gridA, param_gridl]
        param_grid = param_grid[pg]

    # create the classifier
    log_clf0 = LogisticRegressionCV()
    RF_clf0 = log_clf0
    scorers0 = {
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
        'precision_score': make_scorer(precision_score),
        'confusion_matrix': make_scorer(confusion_matrix)
    }
    scorersSens = {
        'recall_score': make_scorer(recall_score)
    }
    scorersAcc = {
        'accuracy_score': make_scorer(accuracy_score)  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
    }
    scorersPrec = {
        'precision_score': make_scorer(precision_score),
        # TP/(TP+FP), a metric of models ability to not miss label a positive
    }
    scorers = [scorersAcc, scorersSens, scorersPrec]
    scr = ['accuracy_score', 'recall_score', 'precision_score']
    s = 0
    cv = 5
    # perform the grid search cross validation
    GSCV_clf0 = GridSearchCV(estimator=RF_clf0, param_grid=param_grid, cv=cv, scoring=scorers[s], refit=scr[s])
    GSCV_clf0.fit(X_tr, y_tr)
    print('Scoring for {:s}'.format(scr[s]))
    print('ZBest Params for set 1:')
    print(GSCV_clf0.best_params_)
    print('best score: ', GSCV_clf0.best_score_)

    RF_clfstd = GSCV_clf0.best_estimator_
    ypz = RF_clfstd.predict(X_ts)
    # fit the
    # RF_clfstd.fit(X_trz, y_train0)

    feature_impz = RF_clfstd.coef_[0]
    # from D_Space import get_current_date
    feates = viz.display_significance(feature_impz, X_tr.columns.values.tolist(), verbose=True)
    pd.DataFrame({'variables': list(feates.keys()), 'Sig': list(feates.values())}).to_excel(
        'Logistic_correlations.xlsx')
    # generate_excel(dic=feates, name='NREL_FEAT_{}_.xlsx'.format(get_current_date()))
    accuracy, scores, posneg, = bi_score(ypz, y_ts, vals=[0,1], classes='')
    print('Sensitivity:', posneg['sen'])
    viz.show_performance(scores=scores, verbose=True)
# =========================================================================
# =========================================================================
#                            numpy tools
# =========================================================================
# =========================================================================
def get_int_mean(na):
    return np.array((np.around(na.mean(axis=0), 0)), dtype=int)

def col_sort(df, col=0):
    return df[df[:, col].argsort()]

def get_select_max_idx(na_row, selection):
    """ find the maximum distance in the given row
            based on the columns (other points) in the selection list
            the idea is that it will find the minimum distance in the
            sample rows row in the distance look up table, to points
            in some other cluster

        :param na_row:  a samples row in the distance look up table, the columns
                        represent the other points in the sample population
        :param selection:  the points you want to look at the distances too,
                           represent points in some other cluster
        :return:        returns the maximum distance and the point that relates to it
        """
    maxi = na_row[selection].max()
    ret = np.where(na_row == maxi)
    ret = ret[0][0]
    return ret, maxi

def get_select_min_idx(na_row, selection):
    """ find the minimum distance in the given row
        based on the columns (other points) in the selection list
        the idea is that it will find the minimum distance in the
        sample rows row in the distance look up table, to points
        in some other cluster

    :param na_row:  a samples row in the distance look up table, the columns
                    represent the other points in the sample population
    :param selection:  the points you want to look at the distances too,
                       represent points in some other cluster
    :return:        returns the minimum distance and the point that relates to it
    """
    mini = na_row[selection].min()
    ret = np.where(na_row == mini)
    ret = ret[0][0]
    return ret, mini
# =========================================================================
# =========================================================================
#                            Usefull math tools
# =========================================================================
# =========================================================================
def mahalanobis_distance(X, mu, cov, is_std=False):
    #print('covariance')
    #print(cov
    xminmu = X - mu
    #print('x - mu')
    #print(xminmu)
    if is_std:
        return np.sqrt((np.dot(xminmu.transpose(), xminmu))/cov)
    return np.sqrt(np.dot(np.dot(xminmu.transpose(), np.linalg.inv(cov)), xminmu))


def euclidean_distance(X, mu, std, is_std=False):
    #print('covariance')
    #print(cov
    xminmu = X - mu
    #print('x - mu')
    #print(xminmu)
    return np.sqrt((np.dot(xminmu.transpose(), xminmu))/std)


def ppm_MSE(ppm1, ppm2):
    # get their covariance matricies
    #print('originals')
    #print(ppm1)
    #print(ppm2)
    cov_1 = pd.DataFrame(ppm1).cov()
    cov_2 = pd.DataFrame(ppm2).cov()
    #print('covariances')
    #print(cov_1.head())
    #print(cov_2.head())
    cov_dif = cov_1 - cov_2
    sum_e = 0
    #for row1, row2 in zip()
    #print('covariance and its square')
    #print(cov_dif.head())
    cov_dif = (cov_dif ** 2)
    #print(cov_dif.head())
    sum = 0
    for row in cov_dif.values:
        sum += row.sum()
    return sum/len(ppm1)


def ppm_AAE(ppm1, ppm2):
    # get their covariance matricies
    #print('originals')
    #print(ppm1)
    #print(ppm2)
    cov_1 = pd.DataFrame(ppm1).cov()
    cov_2 = pd.DataFrame(ppm2).cov()
    #print('covariances')
    #print(cov_1.head())
    #print(cov_2.head())
    cov_dif = cov_1 - cov_2
    sum_e = 0
    #for row1, row2 in zip()

    #print(cov_dif.head())
    cov_dif = np.abs(cov_dif)
    #print(cov_dif.head())
    sum = 0
    for row in cov_dif.values:
        sum += row.sum()
    return sum/len(ppm1)


# =========================================================================
# =========================================================================
#                            Scoring tools
# =========================================================================
# =========================================================================
def bi_score(g, y, vals, classes='', method='accuracy', verbose=False, train=False, retpre=False):
    scores = {'tp':0,
              'fp':0,
              'fn':0,
              'tn':0,}
    posneg = {'bestn':0,
              'bestp':0,
              'predictions':list(g)}

    # go through the guesses and the actual y values scoring
    # * true positives: tp
    # * false positives: fp
    # * true negatives: tn
    # * false negatives: fn
    for gs, ay in zip(g,y.values):
            # check for negative
            if int(gs) == int(vals[0]):
                if int(ay) == int(gs):
                    scores['tn'] += 1
                else:
                    scores['fn'] += 1
            elif int(gs) == int(vals[1]):
                if int(ay) == int(gs):
                    scores['tp'] += 1
                else:
                    scores['fp'] += 1
            else:
                print('Uh Oh!!!!!: {0}'.format(gs))
                print('line number 2461')
                quit(-463)

    posneg['bestn'] = scores['tn']/(scores['fp']+scores['tn'])
    posneg['bestp'] = scores['tp']/(scores['tp']+scores['fn'])

    # calculate and return the overall accuracy
    if method == 'accuracy':
        if retpre:
            accuracy, sum, sensitivity, specificity, precision = viz.show_performance(scores=scores,
                                                                                      verbose=verbose, retpre=retpre)
            posneg['Sensitivity'] = sensitivity
            posneg['Specificity'] = specificity
            posneg['Precision'] = precision
        else:
            accuracy, sum, sensitivity, specificity = viz.show_performance(scores=scores, verbose=verbose,)
            posneg['sen'] = sensitivity
            posneg['spe'] = specificity
        if train:
            return accuracy, scores, posneg
        return accuracy, scores, posneg


# =========================================================================
# =========================================================================
#                Result Recording analysis and documentation
# =========================================================================
# =========================================================================
# *** *** *** *** *** ***     can be used to keep records of tests
class ResultsLog:
    """
        This class can store different results of ML testing
    """
    def __init__(self, result_dict, df_old_log=None, infile_name_old_log=None, sheet_name=None, sort_bys=None,
                 outfile_name_updated_log=None, usecols=None, verbose=False):
        self.result_dict = result_dict
        self.record_name = list(result_dict.keys())           # the names of the attributes
        self.records = list(result_dict.values())             # the values to be added to the log
        self.df_old_log = df_old_log                          # the data frame that contains the logged data if needed, can be left None and will be loaded based on the old log file
        self.infile_name_old_log = infile_name_old_log        # the name of the file to be added to TODO: need to add checker method for file/diretory existence and remove this
        if outfile_name_updated_log is not None:
            self.outfile_name_updated_log=outfile_name_updated_log # TODO: currently neccessary soon to be optional name of new log file if desired
        else:
            self.outfile_name_updated_log=infile_name_old_log # TODO: currently neccessary soon to be optional name of new log file if desired
        self.sheet_name=sheet_name                            # TODO: modify the saving portion to use an excel writer so I can get at specific sheets w/o overwriting the old file
        self.sort_bys=sort_bys                                # optional if you want the log file sorted in a specific way
        self.usecols=usecols                                  #  optional: can select specific columns of the log file to log
        self.verbose=verbose
        if df_old_log is None and (infile_name_old_log is not None):
            self.process_file_name()
        elif df_old_log is not None:
            self.process_df()

    def process_file_name(self,):
        # if the file exists
        if os.path.isfile(self.infile_name_old_log):
            if self.verbose:
                print('The file {} loading...'.format(self.infile_name_old_log))
            if self.usecols is None:
                df_old = pd.read_excel(self.infile_name_old_log)
            else:
                df_old = pd.read_excel(self.infile_name_old_log, usecols=self.usecols)
            df_old = concat_columns(df_old, self.record_name, self.records)
            if self.sort_bys is None:
                df_old.to_excel(self.outfile_name_updated_log, index=False)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(
                    self.outfile_name_updated_log, index=False)
        # if the file does not exist
        else:

            df_old = pd.DataFrame()
            for p, v in zip(self.record_name, self.records):
                df_old[p] = list([v])
            if self.sort_bys is None:
                df_old.to_excel(self.outfile_name_updated_log, index=False)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(
                    self.outfile_name_updated_log, index=False)
            if self.verbose:
                print('The file {} created...'.format(self.infile_name_old_log))

        """ 
        if self.sheet_name is None:
            if self.sort_bys is None:
                dumdf = concat_columns(df_old, self.record_name, self.records)
                dumdf.to_excel(self.outfile_name_updated_log, index=False)
            else:
                dumdf = concat_columns(df_old, self.record_name, self.records)
                dumdf.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(self.outfile_name_updated_log, index=False)
        else:
            if self.sort_bys is None:
                # dumdf = concat_columns(df_old, self.record_name, self.records)
                df_old.to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=True, ascending=False).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
        """

    def process_df(self,):
        if self.sheet_name is None:
            if self.sort_bys is None:
                concat_columns(self.df_old_log, self.record_name, self.records).to_excel(self.outfile_name_updated_log, index=False)
            else:
                concat_columns(self.df_old_log, self.record_name, self.records).sort_values(by=self.sort_bys, inplace=True, ascending=True).to_excel(self.outfile_name_updated_log, index=False)
        else:
            if self.sort_bys is None:
                concat_columns(self.df_old_log, self.record_name, self.records).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
            else:
                concat_columns(self.df_old_log, self.record_name, self.records).sort_values(by=self.sort_bys, inplace=True).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)

# =========================================================================
# =========================================================================
#               TODO: Grid searches
# =========================================================================
# =========================================================================

class GGridSearcher():
    def __init__(self, cmodel, Xtr=None, ytr=None, Xts=None, yts=None, clf=None, param_dict=None, verbose=False,
                 m_type='classifier', make_reports=True, non_prediction=False, attribs=None, model_vars=None,
                 current_model=None, newfile_Per=None, newfile_FI=None, newfile_Re=None, new_tree_png=None):
        if cmodel is not None:
            self.cmodel = cmodel
            self.Xtr = cmodel.X
            self.ytr = cmodel.y
            self.Xts = cmodel.Xts
            self.yts = cmodel.yts
        else:
            self.cmodel = cmodel
            self.Xtr = Xtr
            self.ytr = ytr
            self.Xts = Xts
            self.yts = yts
        self.clf=clf
        self.verbose=verbose
        self.m_type = m_type
        self.param_dict=param_dict
        self.make_reports = make_reports
        self.SkSVCparam_dict = None
        self.GLinRegparam_dict = None
        self.GClstrparam_dict = None
        self.SkKmuparam_dict = None
        self.SkMbKmuparam_dict = None
        self.SKRFparam_dict=None
        self.non_prediction = non_prediction
        self.attribs = attribs
        self.current_model=current_model
        self.model_vars=model_vars
        self.newfile_Per=newfile_Per
        self.newfile_FI=newfile_FI
        self.newfile_Re=newfile_Re
        self.new_tree_png=new_tree_png

    def set_clf(self,clf):
        self.clf=clf
    def set_param_grid(self,param_dict):
        self.param_dict=param_dict
    def set_verbose(self,verbose):
        self.verbose=verbose
    def get_clf(self, ):
        return self.clf
    def get_param_grid(self, ):
        return self.param_dict
    def get_verbose(self, verbose):
        return self.verbose

    def GO(self, report_dict, file=None, sortbys = None, sheet_name=None, usecols=None, ):
        """
            Can be used to run a series of grid search runs for various algorithms.
            The different types are controled by the parameter self.clf
            currently the options are:
                                      * skleranSVC
                                      * sklearnKmu
                                      * sklearnRandomForest

        :param report_dict: a dictionary containing the performance results you want logged
        :param file:  the file you would like to store the report log into
        :param sortbys: the columns you would like to sort the result log by if any
        :param sheet_name: the sheet name of the log file if any TODO: need to create an excel writer method so I can manipulate the sheets
        :param usecols: the columns of the model tested
        :return:
        """
        file = self.newfile_Re
        if self.clf == 'sklearnSVC':
            from sklearn.svm import SVC
            self.SkSVCparam_dict = {'C': [1],
                                    'kernel': ['rbf'],      # kernel type used for algorithm
                                    'degree': [3],          # degree used for polynomial kernel, ignored by all others
                                    'gamma': ['scale'],     # scale (sigma) used in kernel
                                    'coef0': [0],           # bias for poly and sigmoid kernels
                                    'shrk':[True],          # whether to use shrinking huristic
                                    'dsf':['ovr'],          # use one-vs-rest or one vs one (ovo)
                                    'cw':['balanced'],      # weights of different classes (priors)
                                    'prob':[False],         # Whether to enable probability estimates.
                                    'tol':[1e-3],           # Tolerance for stopping criterion.
                                    'max_it':[-1]}         # max allowable iterations
            # change defaults to passed settings
            for pu in self.param_dict:
                if pu in self.SkSVCparam_dict:
                    self.SkSVCparam_dict[pu] = self.param_dict[pu]
            # now do grid search
            C = self.SkSVCparam_dict['C']
            Krnl= self.SkSVCparam_dict['kernel']
            dgr = self.SkSVCparam_dict['degree']
            gma = self.SkSVCparam_dict['gamma']
            co = self.SkSVCparam_dict['coef0']
            shk = self.SkSVCparam_dict['shrk']
            dsf = self.SkSVCparam_dict['dsf']
            clsw = self.SkSVCparam_dict['cw']
            prob = self.SkSVCparam_dict['prob']
            tol = self.SkSVCparam_dict['tol']
            mxit = self.SkSVCparam_dict['max_it']
            for c in C:
                for k in Krnl:
                    for mx in mxit:
                        for s in shk:
                            for df in dsf:
                                for p in prob:
                                    for t in tol:
                                        for cw in clsw:
                                            if k in ['rbf', 'poly', 'sigmoid']:
                                                for g in gma:
                                                    if k in ['poly', 'sigmoid']:
                                                        for coef in co:
                                                            if k is 'poly':
                                                                   for d in dgr:
                                                                       svc_clf = SVC(C=c, kernel=k, degree=d, gamma=g,
                                                                                   coef0=coef, shrinking=s, probability=p,
                                                                                   tol=t, class_weight=cw, max_iter=mx,
                                                                                   decision_function_shape=df)
                                                                       strtm = time.time()
                                                                       svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                                       trpast = time_past(strtm)
                                                                       yp = svc_clf.predict(self.Xts)
                                                                       acc, scr, posneg = bi_score(yp, self.yts, vals=[0,1], retpre=True)
                                                                       if self.make_reports:
                                                                           for r in posneg:
                                                                               if r in report_dict:
                                                                                   report_dict[r] = posneg[r]
                                                                           report_dict['C'] = c
                                                                           report_dict['kernel'] = k
                                                                           report_dict['degree'] = d
                                                                           report_dict['gamma'] = g
                                                                           report_dict['coef0'] = coef
                                                                           report_dict['shrk'] = s
                                                                           report_dict['dsf'] = df
                                                                           report_dict['cw'] = cw
                                                                           report_dict['prob'] = p
                                                                           report_dict['tol'] = t
                                                                           report_dict['max_it'] = mx

                                                                           ResultsLog(report_dict,
                                                                                       infile_name_old_log=file,
                                                                                       outfile_name_updated_log=file,
                                                                                       sheet_name=sheet_name,
                                                                                       usecols=usecols,
                                                                                       sort_bys=sortbys)
                                                            else:
                                                                svc_clf = SVC(C=c, kernel=k, gamma=g,
                                                                              coef0=coef, shrinking=s, probability=p,
                                                                              tol=t, class_weight=cw, max_iter=mx,
                                                                              decision_function_shape=df)
                                                                strtm = time.time()
                                                                svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                                trpast = time_past(strtm)
                                                                yp = svc_clf.predict(self.Xts)
                                                                acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                                if self.make_reports:
                                                                    for r in posneg:
                                                                        if r in report_dict:
                                                                            report_dict[r] = posneg[r]
                                                                    report_dict['C'] = c
                                                                    report_dict['kernel'] = k
                                                                    report_dict['degree'] = -1
                                                                    report_dict['gamma'] = g
                                                                    report_dict['coef0'] = coef
                                                                    report_dict['shrk'] = s
                                                                    report_dict['dsf'] = df
                                                                    report_dict['cw'] = cw
                                                                    report_dict['prob'] = p
                                                                    report_dict['tol'] = t
                                                                    report_dict['max_it'] = mx
                                                                    report_dict['time'] = trpast

                                                                    ResultsLog(report_dict,
                                                                               infile_name_old_log=file,
                                                                               outfile_name_updated_log=file,
                                                                               sheet_name=sheet_name,
                                                                               usecols=usecols,
                                                                               sort_bys=sortbys)
                                                    else: # when rbf
                                                        for g in gma:
                                                            svc_clf = SVC(C=c, kernel=k, gamma=g,
                                                                          shrinking=s, probability=p,
                                                                          tol=t, class_weight=cw, max_iter=mx,
                                                                          decision_function_shape=df)
                                                            strtm = time.time()
                                                            svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                            trpast = time_past(strtm)
                                                            yp = svc_clf.predict(self.Xts)
                                                            acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                            if self.make_reports:
                                                                for r in posneg:
                                                                    if r in report_dict:
                                                                        report_dict[r] = posneg[r]
                                                                report_dict['Accuracy'] = acc
                                                                report_dict['C'] = c
                                                                report_dict['kernel'] = k
                                                                report_dict['degree'] = -1
                                                                report_dict['gamma'] = g
                                                                report_dict['coef0'] = -1
                                                                report_dict['shrk'] = s
                                                                report_dict['dsf'] = df
                                                                report_dict['cw'] = cw
                                                                report_dict['prob'] = p
                                                                report_dict['tol'] = t
                                                                report_dict['max_it'] = mx
                                                                report_dict['time'] = trpast

                                                                ResultsLog(report_dict,
                                                                           infile_name_old_log=file,
                                                                           outfile_name_updated_log=file,
                                                                           sheet_name=sheet_name,
                                                                           usecols=usecols,
                                                                           sort_bys=sortbys)
                                            else:   # if linear
                                                svc = SVC(C=c, kernel=k, )
                                                strtm = time.time()
                                                svc_clf = SVC(C=c, kernel=k, shrinking=s, probability=p,
                                                              tol=t, class_weight=cw, max_iter=mx,
                                                              decision_function_shape=df)
                                                svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                trpast = time_past(strtm)
                                                yp = svc_clf.predict(self.Xts)
                                                acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                if self.make_reports:
                                                    for r in posneg:
                                                        if r in report_dict:
                                                            report_dict[r] = posneg[r]
                                                    report_dict['C'] = c
                                                    report_dict['kernel'] = k
                                                    report_dict['degree'] = -1
                                                    report_dict['gamma'] = -1
                                                    report_dict['coef0'] = -1
                                                    report_dict['shrk'] = s
                                                    report_dict['dsf'] = df
                                                    report_dict['cw'] = cw
                                                    report_dict['prob'] = p
                                                    report_dict['tol'] = t
                                                    report_dict['max_it'] = mx
                                                    report_dict['time'] = trpast

                                                    ResultsLog(report_dict,
                                                               infile_name_old_log=file,
                                                               outfile_name_updated_log=file,
                                                               sheet_name=sheet_name,
                                                               usecols=usecols,
                                                               sort_bys=sortbys)
        elif self.clf == 'sklearnKmu':
            from sklearn.cluster import KMeans as kmu
            self.SkKmuparam_dict = {'n_clusters':[2],
                                    'init':['k-means++'],
                                    'n_init':[10],
                                    'max_iter':[300],
                                    'tol':[1e-4],
                                    'algorithm':['auto',]}
            # add user chosen test sets
            for pu in self.param_dict:
                if pu in self.SkKmuparam_dict:
                    self.SkKmuparam_dict[pu] = self.param_dict[pu]
                # now do grid search
            n_clusters = self.SkKmuparam_dict['n_clusters']
            ini = self.SkKmuparam_dict['init']
            nini = self.SkKmuparam_dict['n_init']
            algo = self.SkKmuparam_dict['algorithm']
            tol = self.SkKmuparam_dict['tol']
            mxit = self.SkKmuparam_dict['max_iter']

            for ncl in n_clusters:
                for i in ini:
                    for n in nini:
                        for al in algo:
                            for tl in tol:
                                for mx in mxit:
                                    KM = kmu(n_clusters=ncl, init=i, algorithm=al, tol=tl, max_iter=mx, n_init=n)
                                    strtm = time.time()
                                    KM.fit(self.Xtr, self.ytr)
                                    trpast = time_past(strtm)
                                    yp = KM.predict(self.Xts)
                                    if ncl == 2:
                                        acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                    else:
                                        acc = metrics.accuracy_score(self.yts, yp)
                                        posneg = {}
                                        posneg['Sensitivity'] = -999
                                        posneg['Specificity'] = -999
                                        posneg['Precision'] = -999
                                    if self.make_reports:
                                        for r in posneg:
                                            if r in report_dict:
                                                report_dict[r] = posneg[r]
                                        report_dict['Accuracy'] = acc
                                        report_dict['Homogeneity'] = metrics.homogeneity_score(self.yts.values.flatten(), yp)
                                        report_dict['n_clusters'] = ncl
                                        report_dict['init'] = i
                                        report_dict['n_init'] = n
                                        report_dict['algorithm'] = al
                                        report_dict['tol'] = tl
                                        report_dict['max_iter'] = mx
                                        report_dict['time'] = trpast

                                        ResultsLog(report_dict,
                                                   infile_name_old_log=file,
                                                   outfile_name_updated_log=file,
                                                   sheet_name=sheet_name,
                                                   usecols=usecols,
                                                   sort_bys=sortbys)
        elif self.clf == 'sklearnRandomForest':
            nruns = 1
            model_vars = self.model_vars
            current_model = self.current_model
            rl = self.attribs
            self.SKRFparam_dict = {
                                    'n_estimators': [2200],  # how many trees in forest
                                    'max_features': [None],  # maximum number of features to test for split
                                    'criterion': ['entropy'],  # how best split is decided
                                    'max_depth': [None],  # how large trees can grow
                                    'oob_score': [True],  #
                                    'warm_start': [True],
                                    'min_samples_leaf': [1],  # The minimum number of samples required to be at a leaf node
                                    'max_leaf_nodes': [None],
                                    'min_weight_fraction_leaf': [0],  #
                                    'min_samples_split': [2],
                                    'min_impurity_decrease': [0],
                                    'random_state': [None],
                                    'class_weight': [None],
                                    'number of warm runs':1
                                }
            for pu in self.param_dict:
                if pu in self.SKRFparam_dict:
                    self.SKRFparam_dict[pu] = self.param_dict[pu]
            warm_start = self.SKRFparam_dict['warm_start']
            if warm_start:
                self.SKRFparam_dict['n_estimators'] = sorted(self.SKRFparam_dict['n_estimators'])
            nruns = self.SKRFparam_dict['number of warm runs']
            for ne in self.SKRFparam_dict['n_estimators']:
                for crit in self.SKRFparam_dict['criterion']:
                    for mxd in self.SKRFparam_dict['max_depth']:
                        for mln in self.SKRFparam_dict['max_leaf_nodes']:
                            RF_clfstd = RandomForestClassifier(n_estimators=ne, criterion=crit, max_depth=mxd,
                                                               warm_start=True, max_leaf_nodes=mln)
                            best_estimator_fit_stime = time.time()
                            for i in range(nruns):
                                RF_clfstd.fit(self.Xtr, self.ytr)
                            best_estimator_fit_etime = time.time() - best_estimator_fit_stime
                            if self.verbose:
                                print("Fitting the best one took {}".format(best_estimator_fit_etime))
                            feature_impz = RF_clfstd.feature_importances_
                            testing_stime = time.time()
                            ypz = RF_clfstd.predict(self.Xts)
                            testing_etime = time.time() - testing_stime
                            feates = display_significance(feature_impz, rl, verbose=True)
                            scores0 = cross_val_score(RF_clfstd, self.Xts, self.yts, cv=2)
                            avg_scr = scores0.mean()
                            print('The Average score set {0}: {0}'.format(0, avg_scr))
                            # score the models performance and show a confusion matrix for it
                            accuracy, scores, posneg, = bi_score(ypz, self.yts, vals=[0, 1], classes='', retpre=True)
                            nwim = self.new_tree_png
                            tmpim = r'C:\Users\gjone\DeepSolar_Code_Base\tree.dot'
                            if nwim is not None:
                                print('creating')
                                print(nwim)
                                viz.display_DT(RF_clfstd.estimators_[0], rl, ['0','1'], newimg=nwim, tmpimg=tmpim,
                                               precision=2)

                            # pd.DataFrame({'variables':list(feates.keys()), 'Sig':list(feates.values())}).to_excel('RandomForest_Feature_significance_18_{}_.xlsx'.format(get_current_date()))
                            # TODO: below line store in generic time and date stamped file
                            # generate_excel(dic=feates, name='RandomForest_Feature_significance_{}_.xlsx'.format(get_current_date()))

                            if self.verbose:
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print('Accuracy: {:.2f}'.format(accuracy))
                                print('Cross val score: {:.3f}'.format(avg_scr))
                                print('Sensitivity:', posneg['Sensitivity'])
                                print('Specificity:', posneg['Specificity'])
                                print('Precision:', posneg['Precision'])
                                viz.show_performance(scores=scores, verbose=True)
                                # print('Training/Testing Split {0}/{1}'.format(tr, ts))
                                print('Training time {}'.format(best_estimator_fit_etime))
                                print('Testing time {}'.format(testing_etime))
                                print('Total time {}'.format(testing_etime + best_estimator_fit_etime))
                                print('Model file ', current_model)
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                            # now save the results dummy
                            params_re = {'Accuracy': accuracy,
                                         'Cross_V2': avg_scr,
                                         'Sensitivity': posneg['Sensitivity'],
                                         'Precision': posneg['Precision'],
                                         'Specificity': posneg['Specificity'], 'runs': 0,
                                         'time': testing_etime + best_estimator_fit_etime}
                            # TODO: fix this file and below as well
                            #new_file = '__Data/__Mixed_models/policy/RF_Perf_{}_.xlsx'.format(
                            #    'DeepSolar_Model_2019-12-30_mega')

                            # store the log files if needed
                            if self.make_reports:
                                for r in posneg:
                                    if r in report_dict:
                                        report_dict[r] = posneg[r]

                                report_dict['Accuracy'] = np.around(accuracy, 3)
                                for r in  self.SKRFparam_dict:
                                    report_dict[r] = self.SKRFparam_dict[r]

                                ResultsLog(report_dict,
                                           infile_name_old_log=file,
                                           outfile_name_updated_log=file,
                                           sheet_name=sheet_name,
                                           usecols=usecols,
                                           sort_bys=sortbys)
                                if self.newfile_Per is not None:
                                    pandas_excel_maker(self.newfile_Per, params_re, mode='performance')
                                # RF_FI = 'RF_FI_{}_.xlsx'.format('DeepSolar_Model_2019-12-30_mega'+'_tc{}xc{}tr{}ts{}')
                                # pandas_excel_maker('__Data/__Mixed_models/policy/RF_FI_{}_.xlsx'.format(model_vars),
                                if self.newfile_FI is not None:
                                    pandas_excel_maker(self.newfile_FI,
                                                       params=feates)

def display_significance(feature_sig, features, verbose=False, reverse=True):
    """

    :param feature_sig: feature significance ranking from RF
    :param features:    list of features
    :param verbose: print?
    :return: sorted dictionary with features as keys and significance as vals
    """
    rd = {}
    for s, f in zip(feature_sig, features):
        rd[f] = s

    sorted_rd = dict(sorted(rd.items(), key=operator.itemgetter(1), reverse=reverse))
    if verbose:
        display_dic(sorted_rd)
    return sorted_rd

def GJ_sklearn_train_test(df, target, trsz=.50, cv=2, rl=None, verbose=True):

    if rl is None:
        rl = rmv_list(df.columns.values.tolist(), target)

    # targets0 = df[target].values.flatten()
    targets0 = df.loc[:, target].values.tolist()
    targets0 = [x[0] for x in targets0]
    print(targets0)
    df = df.loc[:, rl]
    if verbose:
        print(df.describe())
        print()
    ts = .50
    tr = 1 - ts
    # Create training and testing sets for the data
    X_train0, X_test0, y_train0, y_test0 = train_test_split(df, targets0, stratify=targets0, test_size=ts,
                                                            train_size=tr)
    return (X_train0, y_train0), (X_test0, y_test0)

def get_suggested_eta(N, denom=12):
    return N/denom

def get_suggested_perp(N, pct=.01):
    return N * pct

def performance_logger(performance_dict, log_file, verbose=False):
    """
        will store the performance results of some form of testing
    :param performance_dict: dictionary where keys are the metric/parameter, and vals are results
    :param log_file: the file name you want to use to store the results
    :param verbose: how much of the process you want displayed to std out
    :return: None
    """
    # check for file and if not found make it

def process_grid_input():
    lcnq = input("perform lcn?: y/n")
    if lcnq.lower() == 'y':
        lcn_reduce = True  # want to reduce it by correlation filtering?
        gmtc = int(input('minimum target correlation? (-1) for none: '))
        gmxcc = int(input('maximmum predictor cross correlation? (2) for none: '))
    else:
        lcn_reduce = False  # want to reduce it by correlation filtering?
    use_full = input('Use the full model (y) or a select predictor set (n)? (y/n): ')
    if use_full.lower() == 'n':
        use_full = False
        usecols = input('Give me the name of the attrib file: ')
        usecols = pd.read_excel(usecols)['variable'].values.tolist()
    else:
        use_full = True  # do you want to use the full model or select features
        usecols = None     # if allowed to be none will use the drops list
    tssp = float(input('validation set percentage (ex. .50): '))
    current_model = input('Give me the name or path to the model file: ')  # the model file to load
    scaler_ty = 'None'
    #s = 0
    #cv = 3
    #n_est = 2200
    #crit = 'entropy'
    #mx_dth = 20
    #print_tree = True

def load_tree_trunc_features(df=None, dffile=None, limit=.00, verbose=False):
    if df is None:
        df = pd.read_excel(dffile, usecols=['Variable', 'Imp_trunc'])

    df = df.loc[df['Imp_trunc'] >= limit, 'Variable']
    print(list(df))
    return list(df)

def forward_sub2(Train_data, Test_data, feats, clf, verbose=True):
    """performs forward substitution dimension reduction
    :param Train_data: list for X,y of training data
    :param Test_data: list for X,y of testing data
    :param feats: features to test
    :param clf: the classifiery to test, must have a fit method
    :param verbose:
    :return: the list of all vars that lead to increase in performance
    """
    # set up vars
    # need a used up list
    from _products.performance_metrics import calculate_vif, calculate_log_like
    best_scr, BRsqr = 0, 0
    used, good, goodR2, current = list(), list(), list(), list()
    best_R2, BRacc = 0, 0,
    tvar = list(feats[:])
    Rtvar = list(feats[:])
    cadd = None
    better_score = True
    # go through checking each variable one by one
    # subing in values
    while better_score:
        better_score = False
        cadd = None
        # go through each of the remaining vars
        # looking for best result, and adding the one that leads to this
        for var in tvar:
            if var not in good:
                current = good + [var]
                v = clf.fit(Train_data[0].loc[:,current ], Train_data[1])

                if v is not None:
                    print('NEED TO HANDLE THE ISSUE')
                    continue
                # tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=2).mean()
                ts_scr = clf.score(Test_data[0].loc[:, current], Test_data[1], )
                Rsqr = clf.get_Macfadden()
                if verbose:
                    pass
                    #print('current:')
                    #print(current)
                    #print()
                    #print('p-value of {}'.format(var))
                    #print(clf.fitted_model.pvalues[var])
                    #if len(current) > 1:
                    #    vif = calculate_vif(Train_data[0].loc[:,current ])
                    #    print('VIF:\n', vif)
                    #print('# ################################################3')
                    #print('# ################################################3')
                    #print('Anova: ')
                    #print(clf.fitted_model.summary())
                    #print('# ################################################3')
                    #print('# ################################################3')
                if ts_scr > best_scr:
                    if clf.fitted_model.pvalues[var] < .055:
                        better_score=True
                        print(' ******************   p value {:.3f}'.format(clf.fitted_model.pvalues[var]))
                        print(' ******************   New best from {} of {}'.format(var, ts_scr))
                        print(' ******************   Rsquare of {}'.format(Rsqr))
                        best_scr = ts_scr
                        best_R2 = Rsqr
                        cadd = [var]
        if cadd is None:
            print('Good Accuracy list, score: {:.3f}'.format(best_scr))
            print(good)
            #sound_alert_file('sounds/this_town_needs.wav')
            break
        good += cadd
        print('Good is now: score: {}'.format(best_R2))
        #print(good)
        tvar = rmv_list(tvar, cadd[0])

    current = list()
    better_score = True
    while better_score:
        better_score = False
        radd = None
        for var in Rtvar:
            if var not in goodR2:
                current = goodR2 + [var]
                if verbose:
                    pass
                    #print('current list to test:')
                    #print(current)
                    #print()
                clf.fit(Train_data[0].loc[:, current], Train_data[1])
                ts_scr = clf.score(Test_data[0].loc[:, current], Test_data[1], )
                # tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=2).mean()
                if verbose:
                    pass
                    #if len(current):
                    #    vif = calculate_vif(Train_data[0].loc[:,current ])
                    #    print('VIF:\n', vif)
                    #print('# ################################################3')
                    #print('# ################################################3')
                    #print('Anova: ')
                    #print(clf.fitted_model.summary())
                    #print('p-values')
                    #print(clf.fitted_model.pvalues[var])
                    #print('# ################################################3')
                    #print('# ################################################3')
                Rsqr = clf.get_Macfadden()
                if Rsqr > BRsqr and clf.fitted_model.pvalues[var] < .055:
                    # check for significance of model
                    print(clf.fitted_model.pvalues)
                    better_score=True
                    print(' ******************   New best Rsqr {} of {}'.format(var, Rsqr))
                    print(' ******************   Accuracy of {}'.format(ts_scr))
                    print(' ******************   pvalue {:.3f}'.format(clf.fitted_model.pvalues[var]))
                    BRsqr = Rsqr
                    BRacc = ts_scr
                    radd = [var]

        if radd is None:
            better_score=False
            print('Best list for R squared')
            print(goodR2)
            print('Anova: ')
            print(clf.fitted_model.summary())
            sound_alert_file('sounds/this_town_needs.wav')
            break
        goodR2 += radd
        print('GoodR2 is now:')
        print(goodR2)
        Rtvar = rmv_list(Rtvar, radd[0])
    return good, goodR2, [best_scr, best_R2], [BRsqr, BRacc]

def forward_sub(Train_data, feats, clf, cv=2, verbose=False):
    """performs forward substitution dimension reduction
    :param Train_data: list for X,y of training data
    :param Test_data: list for X,y of testing data
    :param feats: features to test
    :param clf: the classifiery to test, must have a fit method
    :param verbose:
    :return: the list of all vars that lead to increase in performance
    """
    # set up vars
    # need a used up list
    best_scr = 0
    used, good, current, acc_l = list(), list(), list(), list()
    acc_inc, best_l = list(), list()
    tvar = list(feats[:])
    cadd = None
    better_score = True
    # go through checking each variable one by one
    # subing in values
    while better_score:
        cadd = None
        better_score = True
        #best_scr = 0
        # go through each of the remaining vars
        # looking for best result, and adding the one that leads to this
        for var in tvar:
            if var not in good:
                current = good + [var]
                if verbose:
                    print('current:')
                    print(current)
                    print()
                # clf.fit(Train_data[0].loc[:,current ], Train_data[1])
                tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=cv).mean()
                if tr_scr > best_scr:
                    better_score=True
                    print(' ******************   New best test from {} of {}'.format(var, tr_scr))
                    best_scr = tr_scr
                    cadd = [var]

        if cadd is None:
            print('returning list')
            print(good)
            sound_alert_file('sounds/this_town_needs.wav')
            return good, best_scr, acc_l, acc_inc
        acc_l.append(best_scr)
        if len(good) == 0:
            acc_inc.append(best_scr)
        else:
            acc_inc.append(best_scr - acc_inc[-1])
        good += cadd
        print('Score: {}, Good is now:'.format(best_scr))
        print(good)
        tvar = rmv_list(tvar, cadd[0])
        # print(tvar)
    return good, best_scr, acc_l, acc_inc

def ML_DATA_SCALER(training, testing, scl_type='_Z_'):
    if scl_type in ['_Z_', '_nrml_']:
        if scl_type == '_Z_':
            sclr = StandardScaler()
        else:
            sclr = MinMaxScaler()
        scl_training = sclr.fit_transform(training)
        scl_testing = sclr.transform(testing)
        return scl_training, scl_testing
    else:
        print('unknown scale type: {}'.format(scl_type))
        print('your options are: _nrml_ for min max, and _Z_ for Z score')

def drop_impute(df, to_drop=(-999, np.inf, '', )):
    if to_drop is not None:
        for to_replace in to_drop:
            df.replace(to_replace, np.nan, inplace=True)
    df.dropna(inplace=True)
    return


def calculate_target_proportions(df, target):
    targets = df[target].tolist()


def taalib_cross_validate_set(df, target, tr=(.75, ), stratify=True, verbose=False):

    if stratify:
        # make sure each set as same target proportions
        cols = df.columns.tolist()
        feats = cols.copy()
        X_tr, X_ts, y_tr, y_ts = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        del feats[feats.index(target)]
        dfr = pd.DataFrame(df)
        classes = set(df[target].values.tolist())
        num_cl = len(classes)
        if verbose:
            print('there are {} classes: {}'.format(num_cl, classes))
        class_dict = dict()
        for cl in classes:
            class_dict[cl] = dfr.loc[dfr[target] == cl]
            n = np.around(len(class_dict[cl])*tr[0])
            X_tr.append(pd.DataFrame(class_dict[cl][feats].values[0:n, :], columns=feats))
            X_ts.append(pd.DataFrame(class_dict[cl][feats].values[n:, :], columns=feats))

            y_tr.append(pd.DataFrame(class_dict[cl][target].values[0:n, :], columns=[target]))
            y_ts.append(pd.DataFrame(class_dict[cl][target].values[n:, :], columns=[target]))


        N = np.around(dfr.shape[0] * tr[0], 0)
        X_tr = dfr.loc[0:N, :][feats]
        y_tr = dfr.loc[0:N, target]

        X_ts = dfr.loc[N:, :][feats]
        y_ts = dfr.loc[N:, target]
        return X_tr, X_ts, y_tr, y_ts

def scale_data_G(Xtr, Xts, scl_type):

    if scl_type is None:
        return Xtr, Xts
    elif scl_type in ['_nrml_', '_Z_', ]:
        if scl_type == '_nrml_':
            sclr = MinMaxScaler()
        elif scl_type == '_Z_':
            # sclr = G_Z_scaler()
            sclr = StandardScaler()
    else:
        print('unknown scale type {}\noptions are: _nrml_ (min max) or _Z_ (z score)'.format(scl_type))
        quit()
    Xtr = sclr.fit_transform(Xtr)
    Xts = sclr.transform(Xts)
    return Xtr, Xts

def generate_training_testing_data(data, target, dtype='dataframe', usecols=None, verbose=False,stratify=True, ts=.5,
                                   scale_ty='_Z_', cols_to_scl=None, impute=True, replace_list=(-999, np.inf, '', )):
    if impute:
        for to_replace in replace_list:
            data.replace(to_replace, np.nan, inplace=True)
        #data.replace(-999, np.nan, inplace=True)
        #data.replace(np.inf, np.nan, inplace=True)
        #data.replace('', np.nan, inplace=True)
        data.dropna(inplace=True)
    if usecols is None:
        usecols = data.columns.tolist()
    if target in usecols:
        predictors = usecols.copy()
        del predictors[predictors.index(target)]
        usecols = predictors
    Training, Testing = cross_val_splitterG(data, rl=usecols, target=target, ts=ts, verbose=verbose, stratify=stratify)
    cols = data.columns.tolist()
    Xtr, ytr = Training[0], Training[1]
    Xts, yts = Testing[0], Testing[1]
    X_tr, y_tr = pd.DataFrame(Training[0], columns=usecols, dtype=np.float64), pd.DataFrame(Training[1],columns=[target], dtype=np.float64)
    X_ts, y_ts = pd.DataFrame(Testing[0], columns=usecols, dtype=np.float64), pd.DataFrame(Testing[1],
                                                                         columns=[target], dtype=np.float64)
    if scale_ty in ['_Z', 'standard']:
        sclr = StandardScaler()
    else:
        sclr = MinMaxScaler()
    if usecols is None:

        X_trs = sclr.fit_transform(X_tr, cols=cols_to_scl)
    else:
        if cols_to_scl is None:
            cols_to_scl = X_tr.columns.tolist()
        #for var in cols_to_scl:
        #    print('var: {}'.format(var))
        X_trs = pd.DataFrame(X_tr[cols_to_scl])
        X_trs = sclr.fit_transform(X_trs)
        X_trs = pd.DataFrame(X_trs, columns=cols_to_scl)
        for var in X_tr.columns:
            if var not in X_trs.columns:
                X_trs[var] = X_tr[var].columns.tolist()
    X_tss = sclr.transform(X_ts)
    #Xtr, Xts = scale_data_G(Xtr, Xts, scl_type=scale_ty)
    return (X_trs, ytr), (X_tss, yts)
class G_Z_scaler:
    def __init__(self, cols=None, scale_type='_Z_',):
        self.scl_type = scale_type
        self.cols = cols
        if self.scl_type in ['_Z_', '_nrml_']:
            if self.scl_type == '_Z_':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
        self.mus = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, df, cols=None):
        if cols is None and self.cols is None:
            self.cols = df.columns.tolist()
        elif cols is None:
            cols = self.cols
        else:
            self.cols = cols
        self.mus = df[cols].mean(axis=0)
        self.max = df[cols].max()
        self.min = df[cols].min()
        self.std = df[cols].std(axis=0)
        self.cols = cols
    def fit_transform(self, df, cols):
        self.fit(df, cols)
        self.cols = cols
        return self.transform(df)
    def transform(self, df):
        """
        for col in self.cols:
            # strip current coll of nan
            tcol = df[col]
            tcol.replace(-999, np.nan, inplace=True)
            tcol.replace(np.inf, np.nan, inplace=True)
            tcol.replace('', np.nan, inplace=True)
            #tcol.dropna(inplace=True)
            if self.scl_type == '_Z_':
                mu =  self.mus
                stdev = self.std
                df[col] = (df[col] - mu)/stdev
            else:
                mx = tcol.max()
                mn = tcol.min()
                df[col] = (df[col] - mn)/(mx-mn)
        """
        df.replace(-999, np.nan, inplace=True)
        df.replace(np.inf, np.nan, inplace=True)
        df.replace('', np.nan, inplace=True)
        cols = self.cols
        if self.scl_type == '_nrml_':
            print('normalizing data in columns:\n{}'.format(self.cols))
            df[cols] = (df[cols] - self.min)/(self.max - self.min)
        else:
            print('Z scaling data in columns:\n{}'.format(self.cols))
            df[cols] = (df[cols] - self.mus)/(self.std)
        print('description of scaled data:')
        print(df.describe())
        return df

def ranked_pi(learner, X, y, features, top, runs=20, verbose=False, random_state=None):


    # make sure what ever top is it doesn't over step its bounds
    lmt = min(abs(top), len(features)) * np.sign(top)

    # make a numpy array of the given features so we can sorte them based on the PI ranking
    features = np.array(features)

    # get the PI from a the passed data (X, y), for the given number of runs
    result = permutation_importance(learner, X, y, n_repeats=runs,
                                    random_state=random_state)
    # order the result importance scores (var: [accuracy_improvement_scores])
    # rank them based on the average importance and get an index list
    # based on that ranking
    perm_sorted_idx = result.importances_mean.argsort()

default_RFR_params = {
            'n_estimators':100,
            'criterion':'mse',
            'max_depth':None,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'min_weight_fraction_leaf':0.0,
            'max_features': None,
            'max_leaf_nodes':None,
            'min_impurity_decrease':0.0,
            'min_impurity_split':None,
            'bootstrap':True,
            'oob_score':False,
            'n_jobs':None,
            'random_state':None,
            'verbose':0,
            'warm_start':False,
            'ccp_alpha':0.0,
            'max_samples':None
        }

default_RF_params = {
        'n_estimators': 100,
        'criterion': 'entropy',  #criterion{“gini”, “entropy”}, default=”gini”
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': None,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'class_weight': None,
        'ccp_alpha': 0.0,
        'max_samples': None
    }

def numerical_encoder(df, col):
    nl = list()
    uniq = set(df[col].values.tolist())
    coding = {u:c for u, c in zip(uniq, range(1, len(uniq)+1))}
    for v in df[col]:
        nl.append(coding[v])
    df[col + '_nc'] = nl
    return col + '_nc'

def numerically_encode_feats(df, feats):
    chn = list()
    for f in feats:
        chn.append(numerical_encoder(df, f))
    return chn

def remove_targets_list(all_vars, targets):
    """
        This removes the targets from the all_vars list
    :param all_vars:
    :param targets:
    :return:
    """
    for rmv in targets:
        del all_vars[all_vars.index(rmv)]
    return

def df_select(df, select):
    rdf = pd.DataFrame()
    for var in df.columns:
        if var in select:
            rdf[var] = df[var].values.tolist()
    return rdf

def strip_strings_dframe_list(df, to_strip=None, ):
    cols = df.columns.tolist()
    removed = list()
    if to_strip is None:
        to_strip = (type(''), )
    for var in df.columns.tolist():
        sub_df = pd.DataFrame(df[var])
        if isinstance(sub_df[var].values[0], to_strip) :
            del cols[cols.index(var)]
            removed.append(var)
    num_df = df[cols].copy()
    othr_df = df[removed].copy()
    return num_df, othr_df

def string_encoding(dfcol, ):
    # get the new values to encode
    to_encode = sorted(list(set(dfcol)))
    cnt = 0
    encoding_map = {}
    for strng in to_encode:
        encoding_map[strng] = cnt
        cnt += 1
    # now make an encoded version of the original column
    return [encoding_map[strng] for strng in dfcol]


def encode_strings_dframe_list(df, to_strip=None, ):
    cols = df.columns.tolist()
    removed = list()
    if to_strip is None:
        to_strip = (type(''), )
    for var in df.columns.tolist():
        sub_df = pd.DataFrame(df[var])
        if isinstance(sub_df[var].values[0], to_strip) :
            df[var] = string_encoding(df[var].tolist())
    return df



def build_RFC(params=None, ):
    # TODO: sklearn page: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if params is None:
        params = default_RF_params
    rfc = RandomForestClassifier(
                                 n_estimators = params['n_estimators'],
                                 criterion = params['criterion'],
                                 max_depth = params['max_depth'],
                                 min_samples_split=params['min_samples_split'],
                                 min_samples_leaf = params['min_samples_leaf'],
                                 min_weight_fraction_leaf = params['min_weight_fraction_leaf'],
                                 max_features=params['max_features'],
                                 max_leaf_nodes=params['max_leaf_nodes'],
                                 min_impurity_decrease=params['min_impurity_decrease'],
                                 min_impurity_split=params['min_impurity_split'],
                                 bootstrap=params['bootstrap'],
                                 oob_score=params['oob_score'],
                                 n_jobs=params['n_jobs'],
                                 random_state=params['random_state'],
                                 verbose=params['verbose'],
                                 warm_start=params['warm_start'],
                                 class_weight=params['class_weight'],
                                 )
    return rfc

# the below class is broken
# TODO: fix or remove this
class DataContainer:
    def __init__(self, target, train_split=.5, src='Xu', region='tva', tset='default',
                 scale_type='_nrml_', usecols = None, reg_d=None, stratify=True):
        from _products._DEEPSOLAR_ import reg_dict, group_labels2
        if reg_d is None:
            reg_d = reg_dict
        self.src = src
        self.region = region
        self.tset = tset
        self.bgs = group_labels2
        self.target, self.train_split = target, train_split
        file_name = reg_d[region]

        if usecols is None:
            self.block_groups = Block_Groups()
            self.block_groups.load_model(group=src, set=tset)
            self.features = self.block_groups.Model
        else:
            self.features = usecols

        print('The features are: {}'.format(self.features))

        self.scaler = None
        # TODO: the below if /else can be reduced to just after the first if
        #       the above if/else makes it so features is never none
        if self.features is not None:
            if target not in self.features:
                use_cols = [target] + self.features
            else:
                print('found the target: {}'.format(target))
                use_cols = self.features
        else:
            use_cols = None
        self.use_cols = use_cols
        # TODO: load data into data frame
        #self.data = pd.read_csv(file_name, usecols=use_cols)
        self.data = smart_table_opener(file_name, usecols=usecols)
        self.orig_data = self.data.copy()
        self.ON = self.data.shape[0]
        self.OD = self.data.shape[1]
        self.data.replace(-999, np.nan, inplace=True)
        self.data.replace(np.inf, np.nan, inplace=True)
        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(inplace=True)
        self.N = len(self.data)
        if use_cols is None:
            self.features = self.data.columns.tolist()
            self.use_cols = self.features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        self.X_data = self.data[self.features]
        self.Y_data = self.data[self.target]
        self.D=self.X_data.shape[1]
        # TODO: generate the training , testing data
        if stratify:
            self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(self.X_data, self.Y_data, stratify=self.Y_data,
                                                                      test_size=train_split, train_size=1 - train_split)
        else:
            self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(self.X_data, self.Y_data,
                                                                          test_size=train_split,
                                                                          train_size=np.around(1 - train_split, 2))
        self.Ntr = len(self.X_tr)
        self.Nts = len(self.X_ts)
        # TODO: if desired scale the data
        if scale_type != "":
            if scale_type == '_nrml_':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.X_tr = self.scaler.fit_transform(self.X_tr)
            self.X_ts = self.scaler.transform(self.X_ts)
        self.Training = (self.X_tr, self.y_tr)
        self.Testing = (self.X_ts, self.y_ts)

class DeepSolar_Model(DataContainer):
    """ Represents the base version of several deepsolar ML tools"""
    reg_dict = {
        'tva': Data_set_paths.tva_set,
        '7 State': Data_set_paths.seven_set,
        '13 State': Data_set_paths.thirteen_set,
        'US': Data_set_paths.US_set,
        'cnvrg': Data_set_paths.US_Convergent_Base,
        'p1': Data_set_paths.paperset,
        'p2': Data_set_paths.paperset2,
        'p3': Data_set_paths.paperset3,
        'p4': Data_set_paths.paperset4,
        'p5': Data_set_paths.paperset5,
        'ltt': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\US_set_all_OMEGA_1_24_21_Base.csv',
        'HghSlr': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_highSolar.xlsx',
        'T3': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_T3.xlsx',
        'Hot': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_HotSpots.xlsx',
        'NT3': r'C:\Users\gjone\DeepLearningDeepSolar\_Data\Mixed\MEGA\DeepSolar_NT3.xlsx',
    }
    '''
    # stores the pathways to specific data sets that cover different regions
    bgs = [
        'dwelling',
        'hh_size',
        'edu',
        'age',
        'income employment',
        'political'
        'dwelling',
        'climate',
        'geography',
        'habit',
        'gender',
        'policy',
        'energy',
        'Xu',
        'ALL',
    ]
    '''
    regions = ['tva', '7 State', '13 State', 'US', 'Paper1']  # used to label the chosen region for analysis
    block_groups_dict = {'demo': 0,  # 0
                         'policy': 1,  # 1,,
                         'physical': 2,  # 2
                         'habit': 3,  # 3,,,
                         'climate': 4,  # 4,,,
                         'geography': 5,  # 5,,,
                         'population': 6,  # 6,
                         'base': 7,  # 7,
                         'edu': 8,  # 8,,
                         'age': 9,  # 9,,,
                         'hh': 10,  # 10,
                         'ct_demo': 11,  # 11,
                         'income': 12,  # 12,
                         'Xu': 13,  # 13
                         'income employment': 14,  # 14
                         'energy': 15,
                         'dwelling': 16,
                         'hh_size': 17,
                         'political': 18,
                         'gender': 19,
                         None: 20,
                         'personal':21,
                         'financial': 22,
                         'suitability': 23,
                         'incentives': 24,
                         'behavior': 25,
                         'interactions': 26,
                         }                         # ties data type labels to an index in the
    block_groups_dict_idx = {'demo': 'Demographic',  # 0
                             'policy': 'Policy',  # 1
                             'physical': 'Physical Characteristics',  # 2
                             'habit': 'Habit',  # 3,
                             'climate': 'Climate',  # 4,
                             'geography': 'Geography',  # 5,
                             'population': 'Population',  # 6,
                             'base': 'Base_Model1',  # 7,
                             'edu': 'Education',  # 8
                             'age': 'Age',  # 9,
                             'hh': 'House Hold',  # 10,
                             'ct_demo': 'CT Demographics',  # 11,
                             'income': 'Income',  # 12,
                             'Xu': 'Xu\'s variables',  # 13
                             'gender': 'Gender',  # 14
                             'income employment': 'Income, Employment, Ownership',  # 15
                             'energy': 'Energy Usage',
                             'dwelling': 'Dwelling Characteristics',
                             'hh_size': 'Household size',
                             'political': 'Political Affiliation',
                             'personal': "Personal Characteristics",
                             'financial': "Financial Resources",
                             'suitability': "Suitability for Adoption",
                             'incentives': "Financial/Policy Incentives",
                             'behavior': "Behavioral Factors",
                             'interactions': "Interactions and State Fixed",
                             None: 'All',
                             }

    def __init__(self, set_type='default', region='US', verbose=False, tr_split=.5, usecols=None, reg_d=None,
                 scale_type='_nrml_', target='Adoption', scale_cols=None, lr=.002, b1=.9, b2=.99, src='', stratify=True):
        super().__init__(target, train_split=tr_split, src=src, region=region, tset=set_type,
                         scale_type=scale_type, usecols=usecols, reg_d=reg_d, stratify=stratify)
        #self.data_container = DataContainer(target=target, train_split=tr_split, src=src, scale_type=scale_type,
        #                                    tset=set_type)

        self.set_type = set_type
        self.region = region
        self.verbose = verbose
        self.scale_type=scale_type
        self.dset_path = self.reg_dict[region]      # load path to starter data set based on region
        print('predictors: {}'.format(self.features))
        # sets variables as what ever usecols or what is returned from the block group object
        self.variables = self.features
        self.data_set = self.data
        self.model=None
        self.learning_rate = lr
        self.beta_1 = b1
        self.beta_2 = b2
        self.optimizer=None
        #self.KerasModelTester = KerasModelTester()

    def generate_optimizer(self, op_type, lr=.002, b1=.9, b2=.99):



        if op_type.lower() == 'SGD'.lower():
            if lr == None:
                lr = .01
            if b1 == None:
                b1 = 0
            if b2 == None:
                b2 = .9
            self.optimizer = optimizers.SGD(learning_rate=lr, momentum=b1)
        elif op_type.lower() == 'Adamax'.lower():
            if lr == None:
                print('the learning rate is none')
                lr = .001
            if b1 == None:
                b1 = .9
            if b2 == None:
                b2 = .999
            self.optimizer = optimizers.adamax(learning_rate=lr, beta_1=b1, beta_2=b2)
        elif op_type.lower() == 'Nadam'.lower():
            if lr == None:
                lr = .001
            if b1 == None:
                b1 = .9
            if b2 == None:
                b2 = .999
            self.optimizer = optimizers.nadam(learning_rate=lr, beta_1 = b1, beta_2=b2)
        elif op_type.lower() == 'Adam'.lower():
            if lr == None:
                lr = .001
            if b1 == None:
                b1 = 0.9
            if b2 == None:
                b2 = .999
            self.optimizer = optimizers.adam(learning_rate=lr, beta_1=b1, beta_2=b2)
        elif op_type.lower() == 'RMSprop'.lower():
            if lr == None:
                lr = .001
            if b1 == None:
                b1 = 0
            if b2 == None:
                b2 = .9
            self.optimizer = optimizers.RMSprop(learning_rate=lr, rho=b2)
        else:
            if lr == None:
                lr = .01
            if b1 == None:
                b1 = 0
            if b2 == None:
                b2 = .9
            self.optimizer = optimizers.SGD(learning_rate=lr, momentum=b1)
        self.learning_rate =lr
        self.b1 = b1,
        self.b2 = b2

    def keras_model_tester(self, model, losses=('mean_absolute_error', 'binary_crossentropy',),
                           optimizers=('SGD',), lrs=(.09, .001, .002), b1s=(.9,), b2s=(.99,), ):
        for loss in losses:
            for optimizer in optimizers:
                pass

class DeepSolar_Auto_Encoder(DeepSolar_Model):

    """ will create a basic auto encoder model for the deepsolar+nrel=convergent data set"""
    def __init__(self, lr=.001, mb=None, src=None, reg='US', target='Adoption', scale_cols=None, scale_type='_Z_',
                 data_set=None, verbose=False, tr_split=.75, b1=.9, b2=.99, features_only=False, set_type='default',
                 usecols=None, reg_d=None, stratify=True):
        super().__init__(set_type=set_type, region=reg, target=target, scale_type=scale_type, reg_d=reg_d,
                         stratify=stratify, verbose=verbose, tr_split=tr_split, lr=lr, b1=b1, b2=b2, src=src,
                         usecols=usecols)
        if features_only:
            # what ever is in data set is used to train AE
            self.data_set = self.data_set[self.variables]
            scale_cols = self.variables
            self.features = self.variables
        else:
            self.features = self.use_cols

        if scale_type is not None:

            scld_data =self.scaler.fit_transform(pd.DataFrame(self.data_set.filter(items=self.variables).copy()) )
            scld_data = pd.DataFrame(scld_data, columns=self.variables)
            # print(scld_data, '\n----: ', target)
            # scld_data[target] = self.data_set[target].values
            self.Training_data = scld_data
        else:
            self.Training_data = self.data_set
            print('Description of data:\n{}'.format(self.Training_data))
        self.input_shape = self.Training_data.shape
        self.variables = self.features
        print('Auto encoding variables: {}'.format(self.variables))
        self.auto_encoder = None
        self.encoder = None
        self.decoder = None
        self.auto_encoder = None
        self.history = None
        self.losses = None
        self.latent_size = None

    def build_encoderT(self, hl_array=(20,), verbose=False,
                       activations=('linear', 'softmax', 'relu', 'tanh', 'sigmoid'), latent_space2=80,
                       latent_space=10, encoder_level=1, layer_strings=None, params_lists=None,
                       names=('input_layer', 'latent_layer', 'reconstruction_layer', 'laten_layerA1')):

        # class level keras sequental model
        self.auto_encoder = Sequential()
        self.latent_size = latent_space
        # create arrays of parameters to feed into the fcl maker
        print('there will be {} latent features generated'.format(latent_space))

        encoder_shallowd = [['BatchNormalization', 'Dense', 'Dense', 'Dense', ],
                           [
                               [(self.input_shape[1],)],            # batch normalization
                               #[latent_space2, activations[3]],     # input layer
                               [latent_space, activations[3]],      # latent space
                               [self.input_shape[1], activations[0]], # Recreation
                           ]
                           ]
        encoder_shallowe = [[ 'BatchNormalization', 'Dense', 'Dense'],
                           [
                               [(self.input_shape[1],)],  # batch normalization
                               # [latent_space2, activations[3]],     # input layer
                               #[latent_space, self.input_shape[1], activations[2]],  # latent space
                               [latent_space, activations[3]],  # latent space
                               [self.input_shape[1], activations[0]],  # Recreation
                           ]
                           ]
        encoder_shallow = [['Dense', 'Dense'],
                           [
                               #[(self.input_shape[1],)],  # batch normalization
                               # [latent_space2, activations[3]],     # input layer
                                [latent_space, self.input_shape[1], activations[0]],  # latent space
                               #[latent_space, activations[3]],  # latent space
                               [self.input_shape[1], activations[1]],  # Recreation
                           ]
                           ]
        encoder_deep1 = [['BatchNormalization', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense', ],
                           [
                               [(self.input_shape[1],)],               # normalization layer
                               [self.input_shape[1], activations[3]],  # first real layer
                               # TODO: Begin Latent representations
                               [latent_space2, activations[2]],
                               [latent_space, activations[2]],
                               [latent_space2, activations[2]],
                               # TODO: End Latent representations
                               [self.input_shape[1], activations[0]],
                           ]
                           ]

        # the encoder level determines if you have a
        # predetermined 1 hidden or  2 hidden layer AE
        # or given as a list of strings to determine each level
        if encoder_level == 1:
            layer_s = encoder_shallow[0]
            params_list = encoder_shallow[1]
        elif encoder_level == 2:
            layer_s = encoder_deep1[0]
            params_list = encoder_deep1[1]
        else:
            layer_s = layer_strings
            params_list = params_lists
        # this will add the passed layers into the model
        # that is stored as a class object
        add_keras_layers(self.auto_encoder, layer_s, params_list)

        # self.encoder.add(Dense(self.input_shape[1], activation=activations[3], input_dim=self.input_shape[1], name=names[0]))
        #self.encoder.add(BatchNormalization(input_shape=(self.input_shape[1],)))
        #self.encoder.add(Dense(self.input_shape[1], activation=activations[3], name=names[0]))
        #self.encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA1'))  # latent space
        #self.encoder.add(Dense(latent_space, activation=activations[2], name=names[1]))  # latent space
        #self.encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA2'))  # latent space
        #self.encoder.add(Dense(self.input_shape[1], activation=activations[0], name=names[2]))
        #
        return self.auto_encoder
    #                                                                      0         1         2        3        4
    def build_encoder(self, hl_array=(20,), verbose=False, activations=('linear', 'softmax', 'relu', 'tanh', 'sigmoid'), latent_space2=80,
                      latent_space=10, names=('input_layer', 'latent_layer', 'reconstruction_layer', 'laten_layerA1')):
        self.auto_encoder = Sequential()
        # create arrays of parameters to feed into the fcl maker
        lyr_Top = [[self.input_shape[1], self.input_shape[1], activations[3]]]
        h1 = [latent_space, 'softmax']

        #self.encoder.add(Dense(self.input_shape[1], activation=activations[3], input_dim=self.input_shape[1], name=names[0]))
        self.auto_encoder.add(BatchNormalization(input_shape=(self.input_shape[1],)))
        self.auto_encoder.add(Dense(self.input_shape[1], activation=activations[3], name=names[0]))
        self.auto_encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA1'))        # latent space
        self.auto_encoder.add(Dense(latent_space, activation=activations[2], name=names[1]))        # latent space
        self.auto_encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA2'))        # latent space
        self.auto_encoder.add(Dense(self.input_shape[1], activation=activations[0], name=names[2]))
        return self.auto_encoder

    def build_encoderB(self, hl_array=(20,), verbose=False, activations=('linear', 'softmax', 'relu', 'tanh', 'sigmoid'), latent_space2=80,
                      latent_space=10, names=('input_layer', 'latent_layer', 'reconstruction_layer', 'laten_layerA1')):
        self.auto_encoder = Sequential()
        # create arrays of parameters to feed into the fcl maker
        lyr_Top = [[self.input_shape[1], self.input_shape[1], activations[3]]]
        h1 = [latent_space, 'softmax']
        self.latent_size = latent_space
        #self.encoder.add(Dense(self.input_shape[1], activation=activations[3], input_dim=self.input_shape[1], name=names[0]))
        self.auto_encoder.add(BatchNormalization(input_shape=(self.input_shape[1],)))
        self.auto_encoder.add(Dense(self.input_shape[1], activation=activations[3], name=names[0]))
        self.auto_encoder.add(Dense(latent_space, activation=activations[2], name=names[1]))        # latent space
        self.auto_encoder.add(Dense(self.input_shape[1], activation=activations[0], name=names[2]))
        return self.auto_encoder

    def compile_encoder(self, opt_type='SGD', lr=.2, b1=.9, b2=.99, loss="binary_crossentropy",
                        metricsl=[accuracy_score,]):
        self.generate_optimizer(op_type=opt_type, lr=lr, b1=b1, b2=b2)
        #self.auto_encoder.compile(optimizer=self.optimizer, loss=loss, metrics=[explained_variance_score(), mean_absolute_error(), mean_squared_error(), r2_score()])
        #self.auto_encoder.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy', 'explained_variance_score',])
        self.auto_encoder.compile(optimizer=self.optimizer, loss=loss, metrics=metricsl)

    def train_encoder(self, epochs=200, batch_size=None):
        self.history = self.auto_encoder.fit(self.data_set, self.data_set, epochs=epochs, batch_size=batch_size)
        self.losses = self.history.history['loss']

    def get_decoder(self, ):
        print('the input shape {}'.format(self.input_shape))
        self.encoder = Sequential()
        #self.encoder.add( tf.keras.Input(shape=(self.input_shape[1], ))  )
        # grab the first layer of the network to use as a decoder
        encoder_layer = self.auto_encoder.layers[0]
        input_data = Input(shape=(self.input_shape[1], ))
        self.encoder = Model(input_data, encoder_layer(input_data))
        #self.encoder = Model(encoder_layer(input_size=self.input_shape))
        self.decoder = self.auto_encoder
        print('\t\t\tEncoder')
        self.encoder.summary()
        print('\t\t\tDecoder')
        self.decoder.summary()


class DeepSolarDenseModel(DeepSolar_Model):
    """ will create a basic Dense model for the deepsolar/nrel data set"""

    def __init__(self, lr=.001, mb=None, src=None, reg='US', target='Adoption', scale_cols=None, scale_type='_Z_',
                 verbose=False, tr_split=.5, b1=.9, b2=.99, features_only=False, set_type='default', reg_d=None):
        super().__init__(set_type=set_type, region=reg, target=target, scale_type=scale_type,
                         verbose=verbose, tr_split=tr_split, lr=lr, b1=b1, b2=b2, src=src, reg_d=reg_d)
        if features_only:
            self.data_set = self.data_set[self.variables]
            scale_cols = self.variables
        if scale_type is not None:
            self.Training_data = self.scaler.fit_transform(self.data_set, cols=scale_cols, )
        else:
            self.Training_data = self.data_set
            print('Description of data:\n{}'.format(self.Training_data))
        self.input_shape = self.Training_data.shape
        self.features = self.Training_data.columns.values.tolist()
        self.model = None
        self.weights_dict = None
        self.history = None
        self.losses = None

class ConvergentSolarAEModel(DeepSolar_Model):

    def __init__(self, src='Xu', reg='US', target='Adoption', scale_type='_Z_', verbose=False, tr_split=.5,
                 lr=.002, b1=.9, b2=.99, set_type='default', auto_params=(None,), auto_dense_params=(None,),
                 dense_params=(None,), features_only=True, l_names={}, params_l={}, comp_params={}, stratify=False,
                 reg_d=None, ):
        #super(ConvergentSolarAEModel, self).__init__(set_type=src, region=reg, target=target, scale_type=scale_type,
        super().__init__(set_type=set_type, region=reg, target=target, scale_type=scale_type,
                         verbose=verbose, tr_split=tr_split, lr=lr, b1=b1, b2=b2, src=src,
                         reg_d=reg_d, stratify=stratify)
        from __Keras_Tools_.keras_tools import KerasVisuals
        self.viz = Visualizer()
        self.k_viz = KerasVisuals()
        self.DS_AE = None
        # TODO: OPTION: 1, an auto_encoder network trained with the target as a feature to learn
        # TODO: OPTION: 2, an auto_encoder network trained with just the features to learn
        self.AEfeatures, self.AEvariables = None, None
        self.AETraining_data = None                 #The entire training set to train the AE
        self.AEinput_shape = None                   #Number of features to reduce
        self.AEhistory = None                       # storage for training results
        self.AElosses = None                        # losses from the training epochs
        self.AEaccuracy = None                      # accuracy from training epochs
        self.lrAE = None                            # learning rate for auto encoder
        self.b1AE = None                            # beta 1 for AE
        self.b2AE = None                            # beta 2 for AE
        self.lossAE = None                          # loss for AE
        self.mbAE = None                            # minibatch for AE
        self.latent_size = None                       # dimension to reduce input to
        self.encoder = None                           # the encoder keras layer
        self.decoder = None                           # decoder keras layer

        # a dense network using the trained auto encoder
        self.dense_encoder=None                     # keras model using AE as encoder layer
        self.DEhistory = None                       # history for AE classifier
        self.DElosses = None                        # losses for AEC
        self.DEaccuracy = None                      # accuracy for AEC
        self.DETs_accuracy = None                   # validation accuracy AEC
        self.lrDE, self.opt_typeDE = None, None     # learning rate and optimizer for AEC

        # regular dens model for comparison
        self.reg_dense = None                       # keras regular dense model
        self.lrDC = None                            # learning rate for dense model
        self.opt_typeDC = None                      # optimizer for DC
        self.lossDC = None                          # loss for DC
        self.DChistory = None                       # history for Dense classifier
        self.DCaccuracy = None                      # accuracy for DC
        self.DCTs_accuracy = None                   # validation accuracy for DC
        self.DCS_loss, self.DETs_accuracy = None, None

        self.DS_AED = None
        self.DS_D = None
        #print('the variables: {}'.format(self.variables))
        #quit(-4177)
        # TODO: Make and train the outo encoder
        self.auto_encoder_ = self.generate_auto_encoder(features_only=features_only, scale_type='_Z_',
                                                        params_l=params_l['AE'], variables=self.variables,
                                                        l_names=l_names['AE'],)
        self.auto_compile_ae(comp_params['AE'])


    # TODO: Auto Encoder layer
    def generate_auto_encoder(self, scale_type, features_only, l_names, params_l, variables=None):
        #DS_AE = DeepSolar_Auto_Encoder(lr=lr, mb=mb, src=src, reg=reg, target=target, scale_cols=scale_cols,
        #                       scale_type=scale_type, features_only=features_only, b1=b1, set_type=set_type)
        from __Keras_Tools_.keras_tools import KerasModelTester, add_keras_layers

        if variables is None:
            self.AEfeatures = self.variables
            self.AEvariables = self.variables
        else:
            self.AEfeatures = variables
            self.AEvariables = variables

        if features_only:
            self.data_set = self.data_set[self.AEvariables]
            self.AEfeatures = self.AEvariables
            params_l[0][1] -= 1
            params_l[0][0] -= 1
            if len(params_l) == 2:
                params_l[1][0] -= 1
            else:
                params_l[2][0] -= 1
        else:
            print('using the target')
            self.AEfeatures = self.use_cols + [self.target]
            self.AEvariables = self.use_cols + [self.target]
        if scale_type is not None:
            self.AETraining_data = self.scaler.fit_transform(self.data_set, )
        else:
            self.AETraining_data = self.data_set
            print('Description of data:\n{}'.format(self.AETraining_data))

        self.latent_size = params_l[0][0]

        self.AEinput_shape = self.AETraining_data.shape
        self.AEvariables = self.AEfeatures
        self.auto_encoder_ = Sequential()
        add_keras_layers(self.auto_encoder_, l_names, params_l)
        print('the thing is {}'.format(self.AEvariables))
        print('the thing is {}'.format(self.AEfeatures))
        return self.auto_encoder_

    def auto_compile_ae(self, params):
        lr=params['lr']
        self.lrAE = lr
        b1 = params['b1']
        self.b1AE = b1
        b2 = params['b2']
        self.b2AE = b2
        opt_type = params['opt']
        self.opt_typeAE = opt_type
        loss = params['loss']
        self.lossAE = loss
        self.compile_auto_encoder(lr, b1, b2, opt_type, loss)

    def compile_auto_encoder(self, lr, b1, b2, opt_type='SGD', loss='mean_square_error'):
        self.generate_optimizer(op_type=opt_type, lr=lr, b1=b1, b2=b2)
        self.auto_encoder_.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy', ])

    def train_auto_encoder(self, AETraining_data, epochs=200, batch_size=None, split=None):
        self.mbAE = batch_size
        self.AEhistory = self.auto_encoder_.fit(AETraining_data, AETraining_data, epochs=epochs, batch_size=batch_size,
                                                validation_split=split,)
        self.AElosses = self.AEhistory.history['loss']
        self.AEaccuracy = self.AEhistory.history['accuracy']

    def analyze_ae_training(self, display_weights=False, rank_and_show_weights=False, show_learning=True):
        if show_learning:
            self.show_auto_encoder_loss()
            self.show_auto_encoder_accuracy()

    def show_auto_encoder_loss(self):
        legend = [self.lossAE + ': {:.3f}'.format(self.AElosses[-1])]
        title = 'AE loss/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                                 self.lrAE, self.opt_typeAE,
                                                                                 self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = self.lossAE
        self.k_viz.multi_plotter([self.AElosses], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)

    def show_auto_encoder_accuracy(self):
        legend = ['Final Accuracy' + ': {:.3f}'.format(self.AEaccuracy[-1])]
        title = 'AE Acc/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                               self.lrAE, self.opt_typeAE,
                                                                               self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = 'Accuracy'
        self.k_viz.multi_plotter([self.AEaccuracy], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)

    # TODO: Turn the auto encoder into encoder and decoder layers for the AE classifier
    def generate_AE_DCLF(self,):
        print('the input shape {}'.format(self.AEinput_shape))
        '''
        # TODO: create a dense AE primed clf
        # TODO: has to have the number of neurons that they encoder takes, 
        #       but with the number of inputs minus the target if it was included
        self.dense_encoder = Sequential()
        self.dense_encoder.add(Dense(self.AEinput_shape[1], input_dim=len(self.variables),
                                     activation='tanh'))
        
        # create the encoder layer/model
        self.encoder = Sequential()
        #self.encoder.add( tf.keras.Input(shape=(self.input_shape[1], ))  )
        encoder_layer = self.auto_encoder_.layers[0]
        input_data = Input(shape=(self.AEinput_shape[1], ))
        self.encoder = Model(input_data, encoder_layer(input_data))
        #self.encoder = Model(encoder_layer(input_size=self.input_shape))
        
        self.decoder = self.auto_encoder_
        self.dense_encoder.add(self.encoder)
        self.dense_encoder.add(Dense(1, activation='sigmoid'))
        print('the encoder fcl')
        print(self.dense_encoder.summary())
        '''

        print('making the ae layer')
        print('the encoder')
        self.generate_E_layer()
        self.encoder.summary()
        print('making the ae_dense classifier')
        self.generate_dense_AE_clf()
        print('the encoder fcl')
        print(self.dense_encoder.summary())

    def generate_E_layer(self):
        print('the input shape {}'.format(self.AEinput_shape))
        # create the encoder layer/model
        self.encoder = Sequential()
        encoder_layer = self.auto_encoder_.layers[0]
        input_data = Input(shape=(self.AEinput_shape[1],))
        self.encoder = Model(input_data, encoder_layer(input_data))
        #print('the encoder')
        #self.encoder.summary()


    # TODO: Auto Encoded dense model
    def generate_dense_AE_clf(self,):
        # TODO: create a dense AE primed clf
        # TODO: has to have the number of neurons that they encoder takes,
        #       but with the number of inputs minus the target if it was included
        self.dense_encoder = Sequential()
        self.dense_encoder.add(Dense(self.AEinput_shape[1], input_dim=len(self.variables),
                                     activation='tanh'))
        self.dense_encoder.add(self.encoder)
        self.dense_encoder.add(Dense(1, activation='sigmoid'))

    def compile_dense_encoder(self, lr, b1, b2, opt_type='SGD', loss='mean_square_error'):
        self.lrDE = lr
        self.opt_typeDE = opt_type
        self.lossDE = loss
        self.generate_optimizer(op_type=opt_type, lr=lr, b1=b1, b2=b2)
        self.dense_encoder.compile(optimizer=self.optimizer, loss=loss, metrics=['mean_square_error', ],)

    def train_dense_auto_encoder(self, Training_data, Testing_data=None, epochs=200, batch_size=None,
                                 ):
        self.mbAE = batch_size
        if Testing_data is None:
            self.DEhistory = self.dense_encoder.fit(Training_data[0], Training_data[1], epochs=epochs,
                                                    batch_size=batch_size, validation_split=self.train_split)
            self.DETS_loss, self.DETs_accuracy = self.dense_encoder.evaluate(Training_data[0], Training_data[1])
        else:
            self.DEhistory = self.dense_encoder.fit(Training_data[0], Training_data[1], epochs=epochs,
                                                    batch_size=batch_size, validation_data=tuple(Testing_data), )
            self.DETS_loss, self.DETs_accuracy = self.dense_encoder.evaluate(Testing_data[0], Testing_data[1])
        self.DElosses = self.DEhistory.history['val_loss']
        self.DEaccuracy = self.DEhistory.history['val_accuracy']
    def analyze_de_training(self, display_weights=False, rank_and_show_weights=False, show_learning=True):
        if show_learning:
            self.show_dense_encoder_loss()
            self.show_dense_encoder_accuracy()
            print()
            print()
            print()
            print('-----------------------------------')
            print('-----------------------------------')
            print('-----------------------------------')
            #print('with Training Accuracy With : %.2f' % (accuracyTR * 100))
            #print('losses', lstr_)
            print('with Testing Accuracy With: %.2f' % (self.DETs_accuracy * 100))
            print('losses', self.DETS_loss)

    def show_dense_encoder_loss(self):
        legend = [self.lossDE + ': {:.3f}'.format(self.DElosses[-1])]
        title = 'DE loss/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                                 self.lrDE, self.opt_typeDE,
                                                                                 self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = self.lossDE
        self.k_viz.multi_plotter([self.DElosses], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)

    def show_dense_encoder_accuracy(self):
        legend = ['Final Accuracy' + ': {:.3f}'.format(self.DEaccuracy[-1])]
        title = 'DE Acc/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                               self.lrDE, self.opt_typeDE,
                                                                               self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = 'Accuracy'
        self.k_viz.multi_plotter([self.DEaccuracy], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)


    # TODO: Regular dense model
    def generate_reg_dense(self,):
        self.reg_dense = Sequential()
        self.reg_dense.add(Dense(len(self.variables), input_dim=len(self.variables),
                                 activation='tanh'))
        self.reg_dense.add(Dense(self.AEinput_shape[1], activation='tanh'))
        self.reg_dense.add(Dense(1, activation='sigmoid'))

    def compile_dense_classifier(self, lr, b1, b2, opt_type='SGD', loss='mean_square_error'):
        self.lrDC = lr
        self.opt_typeDC = opt_type
        self.lossDC = loss
        self.generate_optimizer(op_type=opt_type, lr=lr, b1=b1, b2=b2)
        self.reg_dense.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy', ],)

    def train_dense_classifier(self, Training_data, Testing_data=None, epochs=200, batch_size=None,):
        self.mbDC = batch_size
        if Testing_data is None:
            self.DChistory = self.reg_dense.fit(Training_data[0], Training_data[1], epochs=epochs,
                                                    batch_size=batch_size, validation_split=self.train_split)
            self.DCTS_loss, self.DCTs_accuracy = self.reg_dense.evaluate(Training_data[0], Training_data[1])
        else:
            self.DChistory = self.reg_dense.fit(Training_data[0], Training_data[1], epochs=epochs,
                                                    batch_size=batch_size, validation_data=tuple(Testing_data), )
            self.DCTS_loss, self.DCTs_accuracy = self.reg_dense.evaluate(Testing_data[0], Testing_data[1])
        self.DClosses = self.DChistory.history['val_loss']              # store validation loss
        self.DCaccuracy = self.DChistory.history['val_accuracy']        # store validation accuracy

    def analyze_dc_training(self, display_weights=False, rank_and_show_weights=False, show_learning=True):
        if show_learning:
            self.show_dc_loss()
            self.show_dc_accuracy()
            print()
            print()
            print()
            print('-----------------------------------')
            print('-----------------------------------')
            print('-----------------------------------')
            #print('with Training Accuracy With : %.2f' % (accuracyTR * 100))
            #print('losses', lstr_)
            print('with Testing Accuracy With: %.2f' % (self.DCTs_accuracy * 100))
            print('losses', self.DCTS_loss)

    def show_dc_loss(self):
        legend = [self.lossDC + ': {:.3f}'.format(self.DClosses[-1])]
        title = 'DC loss/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                                 self.lrDC, self.opt_typeDC,
                                                                                 self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = self.lossDC
        self.k_viz.multi_plotter([self.DClosses], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)

    def show_dc_accuracy(self):
        legend = ['Final Accuracy DC' + ': {:.3f}'.format(self.DCaccuracy[-1])]
        title = 'DC Acc/epch: {}--mb: {} lr:{:.3f}, opt:{:s}, ldim: {}'.format(self.src, self.mbAE,
                                                                               self.lrDC, self.opt_typeDC,
                                                                               self.latent_size)
        # title = 'The plot'
        xlabel = 'epochs'
        ylabel = 'Accuracy'
        self.k_viz.multi_plotter([self.DCaccuracy], legend, show=True, title=title, xlabel=xlabel, ylabel=ylabel,
                                 fig_num=None)


class Auto_Encoded_Block_group_classifier:
    def __init__(self, block_groups, params, region, scale_type='_Z_', features_only=False, target='Adoption',
                 tr_split=.5, verbose=False, lr=.0002, b1=.00, b2=.99, epochsAE=[150,], mbAE=200,
                 l_names_dict_dict={}, params_l_dict={}, comp_params_dict={},
                 opt_typeAE='RMSprop', stratify=True):
        from __Keras_Tools_.keras_tools import KerasVisuals
        self.block_groups = block_groups
        self.params = params
        self.stratify = stratify
        self.region = region
        self.target = target
        self.scale_type = scale_type
        self.features_only = features_only
        self.tr_split = tr_split
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.verbose = verbose
        self.viz = Visualizer()
        self.k_viz = KerasVisuals()
        self.AE_Dict = {}
        self.epochsAE = epochsAE
        self.mbAE = mbAE
        self.l_names_dict_dict = l_names_dict_dict
        self.params_l_dict = params_l_dict
        self.comp_params_dict = comp_params_dict
        self.AEmerged_data = None
        self.AEmerged_dataDF = None
        self.full_features = None
        self.AE_X, self.AE_y = None, None
        self.x_TR, self.x_TS, self.y_TR, self.y_TS = None, None, None, None
        # pre train the AE networks
        self.generate_AE_layers()
        # create an appropriately arranged data set to train the network with
        self.generate_dense_classifier()


    def generate_AE_layers(self,):
        reg = self.region
        stratify = self.stratify
        t = self.target
        st = self.scale_type
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        targets = None
        for group in self.block_groups:
            self.AE_Dict[group] = ConvergentSolarAEModel(src=group, reg=reg, target=t, scale_type=st,
                                                         verbose=self.verbose, tr_split=self.tr_split,
                                                         lr=lr, b1=b1, b2=b2, set_type='default',
                                                         auto_params=(None,), auto_dense_params=(None,),
                                                         dense_params=(None,), features_only=self.features_only,
                                                         l_names=self.l_names_dict_dict[group],
                                                         params_l=self.params_l_dict[group],
                                                         comp_params=self.comp_params_dict[group],)


            self.AE_Dict[group].train_auto_encoder(self.AE_Dict[group].AETraining_data,
                                                   epochs=self.epochsAE[group],
                                                   batch_size=self.mbAE, split=.3)
            self.AE_Dict[group].analyze_ae_training(display_weights=False,
                                                    rank_and_show_weights=False,
                                                    show_learning=True)
            self.AE_Dict[group].generate_AE_DCLF()
            print('the data set for group {}'.format(group))
            print('the features for group: {}, {}'.format(group, self.AE_Dict[group].features))
            print(self.AE_Dict[group].orig_data[0:5])
            print(self.AE_Dict[group].scaler.fit_transform(self.AE_Dict[group].orig_data))
            #quit(-4532)
            if self.AEmerged_data is None:
                feats = self.AE_Dict[group].features
                targets = self.AE_Dict[group].orig_data[[self.target]]
                ddf = self.AE_Dict[group].scaler.fit_transform(self.AE_Dict[group].orig_data[feats])
                self.AEmerged_data = ddf #self.AE_Dict[group].orig_data[feats]

                self.full_features = feats
            else:
                feats = self.AE_Dict[group].features
                ddf = self.AE_Dict[group].scaler.fit_transform(self.AE_Dict[group].orig_data[feats])
                self.AEmerged_data = np.concatenate([self.AEmerged_data, ddf], axis=1)
                self.full_features += feats

        print('the full features: {}'.format(self.full_features))
        #print('the targets\n{}'.format(targets.reshape(len(targets), 1)))
        self.AEmerged_dataDF = pd.DataFrame(self.AEmerged_data, columns=self.full_features)
        self.AEmerged_dataDF = pd.concat([targets, self.AEmerged_dataDF], axis=1)
        self.AEmerged_dataDF.replace(-999, np.nan, inplace=True)
        self.AEmerged_dataDF.replace(np.inf, np.nan, inplace=True)
        self.AEmerged_dataDF.replace('', np.nan, inplace=True)
        self.AEmerged_dataDF.dropna(inplace=True)
        # now make the training, validation sets
        self.AE_X = self.AEmerged_dataDF[self.full_features]
        self.AE_y = self.AEmerged_dataDF[self.target]

        if self.stratify:
            self.x_TR, self.x_TS, self.y_TR, self.y_TS = train_test_split(self.AE_X, self.AE_y, stratify=self.AE_y,
                                                                  test_size=self.tr_split, train_size=np.around(1 - self.tr_split, 2))
    def generate_dense_classifier(self,):
        model = Sequential()
        #model.add(Dense(len(self.full_features)+len(self.block_groups), input_dim=len(self.full_features),
        #          activation='linear'))
        #model.add(Concatenate([Input(shape=())]))
        input_layer1 = Input(shape=())

        # now concate the layers
        empty = None
        ll = list()
        cl = list()
        ilr = list()
        inss = None
        for group in range(len(self.block_groups)):
            input_layer1 = Input(shape=(len(self.AE_Dict[self.block_groups[group]].features),))
            ilr.append(input_layer1)
            dl = Dense(self.AE_Dict[self.block_groups[group]].OD)(input_layer1)
            #ll.append(dl)
            cl.append(self.AE_Dict[self.block_groups[group]].encoder(dl))
            #print(self.AE_Dict[self.block_groups[group]].encoder.get_output)
            print(self.AE_Dict[self.block_groups[group]].encoder.summary())
            quit(4590)
            ll.append(self.AE_Dict[self.block_groups[group]].encoder)
            #'''
            if empty is None:
                empty = self.AE_Dict[self.block_groups[group]].encoder
                #ll.append(empty.layers[-1].output)
            else:
                empty = Concatenate([empty, self.AE_Dict[self.block_groups[group]].encoder])
                inss = Concatenate([inss, dl])
                #ll.append(self.AE_Dict[self.block_groups[group]].dense_encoder.layers[-1].output)
            #   empty = concatenate([empty, self.AE_Dict[self.block_groups[group]].dense_encoder])
            #print('the empty: {}'.format(empty))
            #'''



        empty = Model(inputs=ilr, outputs=ll)
        print('the empty: {}'.format(empty))
        #out = Dense(1, activation='tanh', name='Classification_Layer')(empty)
        #new_dclf = Model(cl, out)
        #print(new_dclf)
        #empty = empty(ll)
        #model.add(ll)
        empty = empty(inss)
        final_layers = Dense(1, activation='tanh')(empty)
        model.add(empty)
        model.add(Dense(1, activation='tanh'))
        self.AE_DENSE = model
        self.AE_DENSE.summary()

class block_group_networkAE_Network(DeepSolar_Auto_Encoder):
    """
        will generate a model using an AE block group structure
        given groups are collections of variables
        intend to group them into categories, use softmax latent space to
        train an AE for each group, concatenate into block group encoder layer
        use to train regressor for analysis
        AE's trained using fraction of unexplained variance as the loss function to maximize explained variance
    """
    def __init__(self, src='', reg='US', target='Adoption', scale_type='_Z_', verbose=False, tr_split=.5,scale_cols=None,
                 lr=.002, b1=.9, b2=.99, set_type='default', auto_params=(None,), auto_dense_params=(None,),data_set=None,
                 dense_params=(None,), features_only=True, l_names={}, params_l={}, comp_params={}, stratify=False,
                 reg_d=None, features=None, ):
        super().__init__(lr=lr, mb=None, src=src, reg=reg, target=target, scale_cols=scale_cols, scale_type=scale_type,
            data_set=data_set, verbose=verbose, tr_split=tr_split, b1=b1, b2=b2, features_only=features_only, set_type=set_type,
            usecols=features, reg_d=reg_d, stratify=stratify)
        # get the passed parameters for the auto encoder,
        # AE-dense, and dense networks,
        self.auto_params = auto_params
        self.auto_dense_params = auto_dense_params
        self.dense_params = dense_params

        # get some labels for some fancy
        # net work print outs and such
        self.l_names = l_names
        self.params_l = params_l
        self.comp_params    = comp_params

    def build_encoderT(self, hl_array=(20,), verbose=False, group_dict=None, encoder_des=None,
                       activations=('linear', 'softmax', 'relu', 'tanh', 'sigmoid'), latent_space2=80,
                       latent_space=10, encoder_level=1, layer_strings=None, params_lists=None,
                       names=('input_layer', 'latent_layer', 'reconstruction_layer', 'laten_layerA1')):

        # build the model using the model api so we can concatenate them
        # class level keras sequental model
        self.auto_encoder = Sequential()
        self.latent_size = latent_space
        group_layers = {}
        encoder_shallow = [['Dense', 'Dense'],
                           [
                               # [(self.input_shape[1],)],  # batch normalization
                               # [latent_space2, activations[3]],     # input layer
                               [latent_space, self.input_shape[1], activations[0]],  # latent space
                               # [latent_space, activations[3]],  # latent space
                               [self.input_shape[1], activations[1]],  # Recreation
                           ]
                           ]
        for group in group_dict:
            group_layers[group] = Sequential()
            add_keras_layers(group_layers[group],['Dense', 'Dense'] )
            #group_layers[group].add(Dense(len(group_dict[group]), latent_space, activation=activations[0]))
            #group_layers[group].add(Dense(len(group_dict[group]), latent_space, activation=activations[0]))
        # create arrays of parameters to feed into the fcl maker
        print('there will be {} latent features generated'.format(latent_space))

        encoder_shallowd = [['BatchNormalization', 'Dense', 'Dense', 'Dense', ],
                           [
                               [(self.input_shape[1],)],            # batch normalization
                               #[latent_space2, activations[3]],     # input layer
                               [latent_space, activations[3]],      # latent space
                               [self.input_shape[1], activations[0]], # Recreation
                           ]
                           ]
        encoder_shallowe = [[ 'BatchNormalization', 'Dense', 'Dense'],
                           [
                               [(self.input_shape[1],)],  # batch normalization
                               # [latent_space2, activations[3]],     # input layer
                               #[latent_space, self.input_shape[1], activations[2]],  # latent space
                               [latent_space, activations[3]],  # latent space
                               [self.input_shape[1], activations[0]],  # Recreation
                           ]
                           ]

        encoder_deep1 = [['BatchNormalization', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense', ],
                           [
                               [(self.input_shape[1],)],               # normalization layer
                               [self.input_shape[1], activations[3]],  # first real layer
                               # TODO: Begin Latent representations
                               [latent_space2, activations[2]],
                               [latent_space, activations[2]],
                               [latent_space2, activations[2]],
                               # TODO: End Latent representations
                               [self.input_shape[1], activations[0]],
                           ]
                           ]

        # the encoder level determines if you have a
        # predetermined 1 hidden or  2 hidden layer AE
        # or given as a list of strings to determine each level
        if encoder_level == 1:
            layer_s = encoder_shallow[0]
            params_list = encoder_shallow[1]
        elif encoder_level == 2:
            layer_s = encoder_deep1[0]
            params_list = encoder_deep1[1]
        else:
            layer_s = layer_strings
            params_list = params_lists
        # this will add the passed layers into the model
        # that is stored as a class object
        add_keras_layers(self.auto_encoder, layer_s, params_list)

        # self.encoder.add(Dense(self.input_shape[1], activation=activations[3], input_dim=self.input_shape[1], name=names[0]))
        #self.encoder.add(BatchNormalization(input_shape=(self.input_shape[1],)))
        #self.encoder.add(Dense(self.input_shape[1], activation=activations[3], name=names[0]))
        #self.encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA1'))  # latent space
        #self.encoder.add(Dense(latent_space, activation=activations[2], name=names[1]))  # latent space
        #self.encoder.add(Dense(latent_space2, activation=activations[2], name='latent_layerA2'))  # latent space
        #self.encoder.add(Dense(self.input_shape[1], activation=activations[0], name=names[2]))
        #
        return self.auto_encoder


def annotation_corre(dim, ax, matx, color='r', fontdict=None):
    if fontdict is None:
        fontdict = {
            'family': 'serif',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'bold',
            'size': '5',
        }
    for i in range(dim[0]):
        for j in range(dim[1]):
            text = ax.text(j, i, np.around(matx[i, j],3), ha='center', va='center', color=color, fontdict=fontdict)
    return


def display_weights(weights, feats, laynum, cmap='magma', fig_size=(10, 10), name=''):
    print(len(weights[0]), len(weights[1]))
    print(weights)
    class_dict = {}
    values, labels = list(), list()
    for c in range(len(weights)):
        labels += feats

    weights = weights.reshape(len(weights) * len(weights[0]), 1)
    print(weights)
    plt.figure(figsize=fig_size)
    plt.imshow(weights, cmap=cmap)
    plt.yticks(ticks=list(range(len(labels))), labels=list(labels))
    plt.xticks([])
    plt.ylabel('Ground Truth')
    plt.title('Layer {}: {} weights'.format(laynum, name))
    annotation_corre((len(labels), 1), plt, weights)
    plt.show()


def display_weights2(weights, feats, laynum, cmap='magma', fig_size=(10,10), name=''):
    #print(len(weights[0]), len(weights[1]))
    print(weights)
    class_dict = {}
    labels = list()
    n_lr = weights.shape[1]
    n_in = weights.shape[0]
    print('Each of the {} neurons takes inputs from {} inputs'.format(n_lr, n_in))
    #weights = weights.reshape(len(weights) * len(weights[0]), 1)
    long_w = list()
    for neuron in range(weights.shape[1]):
        for inpt in weights[:, neuron]:
            long_w.append(np.array(inpt))
    long_w = np.array(long_w).reshape(len(long_w), 1)
    print('the stuff: ')
    print(long_w)
    weights = long_w
    #weights = weights.reshape(n_lr * n_in, 1)
    #weights = weights.reshape(len(weights) * len(weights[0]), 1)
    #weights = weights.flatten()
    for c in range(n_lr):
        if feats is None:
            labels += ['n{}_'.format(c)+str(f) for f in range(n_in)]
        else:
            labels += ['n{}_'.format(c) + '_' + str(f) for f in feats]
    #print(weights)
    print(labels)
    plt.figure(figsize=fig_size, dpi=200)
    plt.imshow(weights, cmap=cmap)
    plt.yticks(ticks=list(range(len(labels))), labels=list(labels))
    plt.colorbar()
    plt.xticks([])
    plt.ylabel('')
    plt.title('Layer {}: {} weights'.format(laynum, name))
    annotation_corre((len(weights), 1), plt, weights)
    plt.show()


def train_and_test_clf(clf, Training, Testing, verbose=False):
    clf.fit(Training[0], Training[1])
    yptr = clf.predict(Training[0])
    ypts = clf.predict(Testing[0])
    return yptr, ypts


def train_and_test_clf_RF(clf, Training, Testing, verbose=False):
    yptr, ypts = train_and_test_clf(clf, Training, Testing, verbose)
    feature_impz_ = clf.feature_importances_
    return yptr, ypts, feature_impz_


class DataProcessor:
    def __init__(self, file_name, target, features=None, train_split=.5, BG=None, scale_type='_nrml_'):
        self.file_name, self.target, self.features, self.train_split = file_name, target, features, train_split
        self.block_groups = BG
        use_cols = None
        self.scaler = None
        #
        if features is not None:
            use_cols = [target] + features
        else:
            use_cols = None

        print('Features: {}'.format(self.features))
        # TODO: load data into data frame
        self.data = pd.read_csv(file_name, usecols=use_cols)
        self.ON = self.data.shape[0]
        self.data.replace(-999, np.nan, inplace=True)
        self.data.replace(np.inf, np.nan, inplace=True)
        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(inplace=True)
        self.N = len(self.data)
        if use_cols is None:
            self.features = self.data.columns.tolist()
        if self.target in self.features:
            del self.features[self.features.index(self.target)]

        self.X_data = self.data[self.features]
        self.Y_data = self.data[self.target]
        self.D = self.X_data.shape[1]

        # TODO: generate the training , testing data
        self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(self.X_data, self.Y_data, stratify=self.Y_data,
                                                                      test_size=train_split, train_size=1 - train_split)
        # TODO: if desired scale the data
        if scale_type != "":
            if scale_type == '_nrml_':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.X_tr = self.scaler.fit_transform(self.X_tr)
            self.X_ts = self.scaler.transform(self.X_ts)




def calculate_null_score(y_tr, num_runs=100):
    """
                Can be used to calculate the accuracy of 100
                random guesses of a given target set
    :param y_tr:
    :param num_runs:
    :return:
    """
    null_score = list()
    for i in range(num_runs):
        y_null = np.array(y_tr)
        np.random.shuffle(y_null)
        null_score.append(accuracy_score(y_tr, y_null))
    return np.mean(null_score)


def generate_neuron_labels(number_neurons, number_inputs, features):
    labels = list()
    for c in range(number_neurons):
        if features is None:
            labels += ['n{}_'.format(c) + str(f) for f in range(number_inputs)]
        else:
            labels += ['n{}_'.format(c) + '_' + str(f) for f in features]
    return labels


def process_model_weights(model, features, threshold=.01):
    lcnt = 1
    layer_w_dict = {}

    for layer in model.layers:
        if len(layer.get_weights()) == 0:
            continue
        weights = np.array(layer.get_weights()[0])
        labels = list()
        neurons_in_lr = weights.shape[1]
        inputs_to_lr = weights.shape[0]
        layer_w_dict[lcnt] = dict()
        print('Each of the {} neurons takes inputs from {} inputs'.format(neurons_in_lr, inputs_to_lr))
        long_w = list()

        if lcnt > 1:
            features=None

        # TODO: set up the neuron label
        labels = generate_neuron_labels(neurons_in_lr, inputs_to_lr, features)
        #for c in range(neurons_in_lr):
        #    if features is None:
        #        labels += ['n{}_'.format(c) + str(f) for f in range(inputs_to_lr)]
        #    else:
        #        labels += ['n{}_'.format(c) + '_' + str(f) for f in features]
        c = 0
        # look at the weights for the neurons in the layer
        for neuron in range(weights.shape[1]):
            layer_w_dict[lcnt][neuron] = dict()
            for inpt in weights[:, neuron]:
                if np.around(np.abs(inpt), 2) >= threshold:
                    long_w.append(np.array(inpt))
                #layer_w_dict[lcnt][neuron][labels[c]] = np.around(np.abs(inpt), 3)
                    layer_w_dict[lcnt][neuron][labels[c]] = np.around(inpt, 3)
                c += 1
            layer_w_dict[lcnt][neuron] = sort_dict(layer_w_dict[lcnt][neuron], )
        lcnt += 1
    return layer_w_dict


def feature_importance_logger(params, X, Y, usecols, runs=10, label_conversion=None):
    """  This will run a given number of runs of training a random forest and storing
         the feature importance's in a list, generates the random forest based on the parameters in the
         parameter argument
    :param params: dictionary for parameters of random forest
    :param X:
    :param Y:
    :param usecols:
    :param runs:
    :return:
    """
    # below dictioanry will store the names of the features as keys, and lists for vals
    if label_conversion is None:
        importance_storage = {feature:list() for feature in usecols}
    else:
        importance_storage = {label_conversion[feature]:list() for feature in usecols}
    # adjust parameters of RF based on user input
    params_to_use = default_RF_params.copy()
    for param in params:
        params_to_use[param] = params[param]
    params = params_to_use

    for r in range(runs):
        RFC = build_RFC(params)
        RFC.fit(X, Y)
        feature_importance = RFC.feature_importances_
        feats = display_significance(feature_importance, usecols, verbose=False, reverse=False)
        #for feature in importance_storage:
        for feature in feats:
            imp = feats[feature]
            if label_conversion is not None:
                feature = label_conversion[feature]
            importance_storage[feature].append(imp)
    yp = RFC.predict(X)
    viz.plot_confusion_matrix(Y, yp, classes=['NA', 'A'],
                              title='Confusion Matrix Random Forest Ptester')
    # the the averages for each feature
    avg_importance = sort_dict({feat:np.mean(importance_storage[feat]) for feat in importance_storage}, reverse=False)


    #print('the sorted average importance')
    #print(avg_importance)
    feature_list = [importance_storage[feat] for feat in avg_importance]
    return feature_list, list(avg_importance.keys())


def result_file_logger(dic, file, ):
    # use file name to open storage d4ata file if it exists
    # if not create an empty version of it.
    pass

def feature_importance_logger_df_mixer(params, df, usecols, runs=10, target='Adoption', label_conversion=None, tr=(.5, ),
                                       special=False, verbose=False, all_in=True, model='classifier', plot_fi_stat=True,
                                       justify=False):
    """  This will run a given number of runs of training a random forest and storing
         the feature importance's in a list, generates the random forest based on the parameters in the
         parameter argument
    :param params: dictionary for parameters of random forest
    :param X:
    :param Y:
    :param usecols:
    :param runs:
    :return:
    """
    # below dictionary will store the names of the features as keys, and lists for vals
    importance_storage = {feature: list() for feature in usecols}
    if label_conversion is None:
        importance_storage = {feature: list() for feature in usecols}
    else:
        importance_storage = {label_conversion[feature]:list() for feature in usecols}
        #print(importance_storage)
    # adjust parameters of RF based on user input
    if model == 'regression':
        params_to_use = default_RFR_params.copy()
        dict_scores = {
            'mse': list(),
            'mae': list(),
            'cod': list(),
            'R2': list(),
        }
        avg_scores = {
            'mse': 0,
            'mae': 0,
            'R2': 0,
        }
    else:
        params_to_use = default_RF_params.copy()
        dict_scores = {
            'acc': list(),
            'sen': list(),
            'prec': list(),
            'R2': list(),
        }
        avg_scores = {
            'acc': 0,
            'sen': 0,
            'prec': 0,
            'R2': 0,
        }

    for param in params:
        params_to_use[param] = params[param]
    params = params_to_use

    print('the cols ', df.columns.tolist())
    print('model type: {}'.format(model))
    #print(df.columns.tolist())
    #print(df.shape)
    rf_clf = RandomForest_Analzer(params=params, model_type=model)

    variable_rankings_list = {v:[] for v in list(importance_storage.keys())}

    for r in range(runs):
        performance_str = ''
        # RFC = build_RFC(params)
        # rf_clf = RandomForest_Analzer(params=params, model_type=model)
        cm = RMODEL(dataframe=df, columns=usecols, target=target, trtsspl=tr, justify=justify)
        #RFC.fit(cm.Xtr, cm.ytr)
        if not all_in:
            rf_clf.fit(x=cm.Xtr, y=cm.ytr, )
            rf_clf.score(X=cm.Xts, y=cm.yts)
            print('Run: {}/{}\n\t:=Testing: {}, Training: {}, Orig: {}/{}'.format(r + 1, runs, cm.Xts.shape[0], cm.Xtr.shape[0], cm.X.shape[0],
                                                                    cm.Xts.shape[0] + cm.Xtr.shape[0]))
            print("\t:=Training avg: {}, Testing avg: {}".format(cm.ytr.mean()[0], cm.yts.mean()[0]))

        else:
            rf_clf.fit(x=cm.X, y=cm.Y, )
            rf_clf.score(X=cm.X, y=cm.Y)
            # store the different scoring metrics for this run

        #for v in list(variable_rankings_list.keys()):
        for v in list(rf_clf.feature_ranking_storage.keys()):
            if label_conversion is not None:
                cv = label_conversion[v]
            else:
                cv = v
            variable_rankings_list[cv].append(rf_clf.feature_ranking_storage[v])

        # store the different scoring metrics for this run
        for pm in list(rf_clf.score_dict.keys()):
            dict_scores[pm].append(rf_clf.score_dict[pm])
            performance_str += '{}: {:.3f} '.format(pm, rf_clf.score_dict[pm])
        print('\tPerformance: {}'.format(performance_str))
        print('------------------------------------------------')
        #feature_importance = rf_clf.feature_importances_
        #feats = display_significance(feature_importance, usecols, verbose=False, reverse=False)
        #for feature in importance_storage:
        for feature in rf_clf.FI:
            imp = rf_clf.FI[feature]
            if label_conversion is not None:
                feature = label_conversion[feature]
            importance_storage[feature].append(imp)

    # now get the average for all scoreing metrics
    for pm in list(dict_scores.keys()):
        avg_scores[pm] = np.mean(dict_scores[pm])
    variable_rankings_list_std = {}
    variable_rankings_list_mu = {}
    #variable_rankings_list_std = {}

    for v in list(variable_rankings_list.keys()):
        variable_rankings_list_std[v] = [np.std(variable_rankings_list[v])]
        variable_rankings_list_mu[v] = [np.mean(variable_rankings_list[v])]

    # the the averages for each feature
    #if label_conversion is None:
    avg_importance = sort_dict({feat:np.mean(importance_storage[feat]) for feat in importance_storage}, reverse=False)
    feature_list = [importance_storage[feat] for feat in avg_importance]
    #else:
    #    avg_importance = sort_dict({label_conversion[feat]:np.mean(importance_storage[feat]) for feat in importance_storage}, reverse=False)
    #    feature_list = [importance_storage[feat] for feat in avg_importance]
    #print('the sorted average importance')
    #print(avg_importance)

    if special:
        if plot_fi_stat:
            return feature_list, list(avg_importance.keys()), cm, rf_clf.learner, avg_scores, \
                   pd.DataFrame(variable_rankings_list_std), pd.DataFrame(variable_rankings_list_mu)

        return feature_list, list(avg_importance.keys()), cm, rf_clf.learner, avg_scores



    return feature_list, list(avg_importance.keys())



def make_me_a_box(box_data, use_cols, title='the box plot title', fontdict=None,
                  figsize=(20,20), savefigure=False, figname='Box_Plot',
                  format='svg', mainfsize=12):
    font = {'size': mainfsize,
            'weight': 'bold'}
    matplotlib.rc('font', **font)
    if fontdict is None:
        fontdict = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '15',
        }
    fig, ax = plt.subplots(1, 1, figsize=figsize,)
    ax.boxplot(box_data, vert=False,
                labels=use_cols, )
    # ax2.set_title('Top {} features for Permutation Importance '.format(-1 * lmt), x = -0.02)
    ax.set_title(title, fontdict=fontdict,)
    #plt.title(title, fontdict=fontdict)
    ax.set_yticklabels(use_cols, fontdict=fontdict)
    fig.tight_layout()
    if savefigure:
        plt.savefig(figname, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=format,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()


def remove_target_confounders(target, targets_possibles):
    """
        the idea is to remove the target from the target_possibles list
    :param target:
    :param targets_possibles:
    :return: the list minus the target
    """
    if target in targets_possibles:
        del targets_possibles[targets_possibles.index(target)]
    return targets_possibles


def data_cleaner(df, missing_threshold=.20, remove=False):
    """
        This method can be used to remove variables that have missing entries
        above a certain missing_threshold, if remove is left false (default) the
        method only returns a list of those variables that are above the threshold
    :param df:
    :param missing_threshold: (fraction ([0,1])) highest missing percantage
    :param remove:
    :return:
    """
    dfdes = df.describe()
    missing_dict = {'vars': [],
                    'missing': []
                    }
    remove_l = list()
    m_d = {}
    for v in df.columns.tolist():
        if v in dfdes.columns.tolist():
            # make sure what you are looking for is there
            m_d[v] = (df.shape[0] - dfdes.loc['count', v]) / df.shape[0]
            missing_dict['vars'].append(v)
            missing_dict['missing'].append( m_d[v])
            if missing_dict['missing'][-1] >= missing_threshold:
                remove_l.append(v)
    missing_dict = sort_dict(m_d, reverse=True)
    if remove:
        df.drop(remove_l, inplace=True)
    return remove_l, missing_dict

# TODO: Pandas manipulation tools
def basic_drop_impute(df, replace=('', np.inf, -999), inplace=True):
    """
        This method is designed to remove the nan and the passed replace values
    :param df:
    :param replace:
    :param inplace:
    :return:
    """
    for to_replace in replace:
        df.replace(to_replace, np.nan, inplace=True)
    if inplace:
        df.dropna(inplace=True)
        return
    else:
        return df.dropna()

def drop_cols(df, drops, inplace=True, verbose=False):
    if inplace:
        print('Droping the columns: {}'.format(drops))
        df.drop(columns=drops)
        return
    else:
        return df.drop(columns=drops, inplace=False)


def check_table_type(table_name):
    """
    This will return what type of file name you have given it in the form of a string
    that will be either csv for a CSV file, or xlsx for an excel file. This method is used
    by other methods to know what type of file it is working with. If the type of file is
    not one of the two the method terminates you program with a explanation
    :param table_name:  Name of data file to open.
    :return: 'csv' for a CSV, and 'xlsx' for an excel workbook
    """
    if table_name[-3:] == 'csv':
        return 'csv'
    elif table_name[-4:] == 'xlsx':
        return 'xlsx'
    else:
        print('Unknown Table storage type for file: {}'.format(table_name))
        print('Terminating program')
        quit()

def show_missing(df, ):
    missingsO = df.isna().sum()
    for cc in sorted(missingsO.index.tolist()):
        if missingsO[cc] > 0:
            print('{}: {}'.format(cc, missingsO[cc]))
    return

def smart_table_opener(table_file, usecols=None,):
    """
        This will open a dataframe from the given file as long as it is a csv or excel file
    :param table_file: name of table to open
    :param usecols: (optional) the columns of the table you want to load
    :return:
    """
    table_type_options = ['xlsx', 'csv', ]
    if check_table_type(table_file) == table_type_options[0]:
        return pd.read_excel(table_file, usecols=usecols, )
    elif check_table_type(table_file) == table_type_options[1]:
        return pd.read_csv(table_file, usecols=usecols, low_memory=False,)