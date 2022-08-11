import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#-- Read in data --

# Declare an empty list to store each line
lines = []
# Open communities.names for reading text data.
with open ('communities.names', 'rt') as attributes:
    for line in attributes:
        #split each line and keep 2nd element
        lines.append(line.split()[1])

# Read in communities data
df = pd.read_csv('communities.data',header=None)
# Add column names
df.columns = lines
#check size of dataframe
print(df.shape)
# Check first 10 rows
print(df.head(10))
# Check info of dataset
print(df.info(verbose=True))
# replace ? values with numpy nan
df = df.replace('?',np.nan)
# Find the number of columns with missing values
print(df.isnull().any().sum(axis=0))
#check % null values in each column
missing = df.isnull().sum()/(len(df))*100
print(missing)

# Plot of features with nulls
plt.figure(0)
missing[missing > 0].plot.barh()
plt.ylabel('Feature')
plt.xlabel('Percent')
# plt.xticks(rotation=80)
plt.savefig('missing.png', dpi=300, bbox_inches='tight')

# Histogram of target
plt.figure(1)
plt.hist(df['ViolentCrimesPerPop'])
plt.xlabel('Number of Violent Crimes Per 100k Population')
plt.title('Histogram of Violent Crimes')
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')

#check columns with over 50% missing values
print(missing.where(missing>50))
#check for total null values in df
print(df.isnull().sum().sum())
#drop non predictive fields
df = df.drop(columns =['state','county','communityname','community','fold',])

plt.figure(2)
plt.plot(df['ViolentCrimesPerPop'], 'o')
plt.title('Violent Crimes Per 100k')
plt.ylabel('Total number of violent crimes per 100K population')
plt.xlabel('Community index')
plt.savefig('violent.png')

# Set X and y
X = df.drop('ViolentCrimesPerPop', axis=1)
y = df['ViolentCrimesPerPop']

# Impute missing values
imp = IterativeImputer(max_iter=10, random_state=0)
Ximpute = imp.fit_transform(X)
Ximpute = pd.DataFrame(Ximpute, columns = X.columns)

def modelevalgs(X,y,filename):
    """
    evaluate models through grid search

    :param X: dataframe of input variables
    :param y: dataframe of target variable
    :param filename: name of file for output table
    :return: table with hyperparamters for each model with lowest error squared
    """

    #drop any null values
    X = X.dropna(axis=1)

    #train test spit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(X_train.shape, X_test.shape)

    def findCorrelations(correlations, cutoff=0.9):
        """
        Find highly correlated features

        :param correlations: correlation matrix
        :param cutoff: max correlation

        :return: highly correlated features
        """
        corr_mat = abs(correlations)
        varnum = corr_mat.shape[1]
        original_order = np.arange(0, varnum + 1, 1)
        tmp = corr_mat.copy(deep=True)
        np.fill_diagonal(tmp.values, np.nan)
        maxAbsCorOrder = tmp.apply(np.nanmean, axis=1)
        maxAbsCorOrder = (-maxAbsCorOrder).argsort().values
        corr_mat = corr_mat.iloc[list(maxAbsCorOrder), list(maxAbsCorOrder)]
        newOrder = original_order[list(maxAbsCorOrder)]
        del (tmp)
        deletecol = np.repeat(False, varnum)
        x2 = corr_mat.copy(deep=True)
        np.fill_diagonal(x2.values, np.nan)
        for i in range(varnum):
            if not (x2[x2.notnull()] > 0.9).any().any():
                print('No correlations above threshold')
                break
            if deletecol[i]:
                continue
            for j in np.arange(i + 1, varnum, 1):
                if (not deletecol[i] and not deletecol[j]):
                    if (corr_mat.iloc[i, j] > cutoff):
                        mn1 = np.nanmean(x2.iloc[i,])
                        mn2 = np.nanmean(x2.drop(labels=x2.index[j], axis=0).values)
                        if (mn1 > mn2):
                            deletecol[i] = True
                            x2.iloc[i, :] = np.nan
                            x2.iloc[:, i] = np.nan
                        else:
                            deletecol[j] = True
                            x2.iloc[j, :] = np.nan
                            x2.iloc[:, j] = np.nan
        newOrder = [i for i, x in enumerate(deletecol) if x]

        return (newOrder)

    #remove highly correlated features
    to_remove = findCorrelations(X_train.corr())
    X_train = X_train.drop(X_train.columns[to_remove], axis=1)
    X_test = X_test.drop(X_test.columns[to_remove], axis=1)

    #--MLP Grid Search--

    piped = Pipeline([('mlp', MLPRegressor(random_state=1))])
    max_neurons = 120
    max_layers = 5
    params = []
    for i in np.arange(2, max_layers + 1, 1):
        for j in np.arange(5, max_neurons + 1, 5):
            out = tuple(np.repeat(j + 1, i + 1))
            params.append(out)

    mlpparam = [{'mlp__activation': ['logistic', 'tanh', 'relu'],
                   'mlp__hidden_layer_sizes': params}]

    mlpgrid = GridSearchCV(estimator=piped,
                      param_grid=mlpparam,
                      scoring='neg_mean_squared_error',
                      verbose=2,
                      refit = True,
                      cv=5)

    mlpgrid = mlpgrid.fit(X_train, y_train)

    mlp_pred = mlpgrid.predict(X_train)

    mlpmse = mean_squared_error(y_train, mlp_pred)

    mlpr2 = r2_score(y_train, mlp_pred)

    mlpscore = mlpgrid.best_score_

    mlpparams = mlpgrid.best_params_


    #--SVM Grid Search--
    svmr = svm.SVR()

    svmparam = [{'kernel':('linear','poly','rbf','sigmoid'),
                'C':(np.arange(1,10,1)),
                'epsilon':(np.arange(.1,1.1,0.1))}]

    svmgrid = GridSearchCV(estimator=svmr,
                           param_grid=svmparam,
                           scoring='neg_mean_squared_error',
                           verbose=2,
                           refit = True,
                           cv=5)

    svmgs = svmgrid.fit(X_train, y_train)

    svm_pred = svmgrid.predict(X_test)
    svmmse = mean_squared_error(y_test, svm_pred)
    svmr2 = r2_score(y_test, svm_pred)
    
    svmscore = svmgrid.best_score_
    svmparams = svmgrid.best_params_

    #--DTR Grid Search--

    dtr = DecisionTreeRegressor()
    #initially set low to ensure that code will run without errors
    dtrparam = [{'max_features':['auto','sqrt'],
                 'max_depth':(np.arange(10,65,1)),
                  }]

    dtrgrid = GridSearchCV(estimator=dtr,
                           param_grid=dtrparam,
                           scoring='neg_mean_squared_error',
                           verbose=2,
                           refit=True,
                           cv=5)

    dtrgs = dtrgrid.fit(X_train, y_train)

    dtr_pred = dtrgrid.predict(X_test)
    dtrmse = mean_squared_error(y_test, dtr_pred)

    dtrr2 = r2_score(y_test, dtr_pred)

    dtrscore = dtrgrid.best_score_
    dtrparams = dtrgrid.best_params_

    #--RFR Grid Search--
    rfr = RandomForestRegressor()

    rfrparam = [{'n_estimators':(np.arange(50,201,10)),
                 'max_features':['sqrt'],
                 'max_depth':(np.arange(10,51,5)),
                 'bootstrap':[True,False]
                }]

    rfrgrid = GridSearchCV(estimator=rfr,
                           param_grid=rfrparam,
                           scoring='neg_mean_squared_error',
                           verbose=2,
                           refit=True,
                           cv=5)
    rfrgs = rfrgrid.fit(X_train, y_train)

    rfr_pred = rfrgrid.predict(X_test)
    rfrmse = mean_squared_error(y_test, rfr_pred)

    rfrr2 = r2_score(y_test, rfr_pred)
    
    rfrscore = rfrgrid.best_score_
    rfrparams = rfrgrid.best_params_

    #--Evaluation Table--
    evaltable = pd.DataFrame({
        'Model': ['SVR', 'DTR', 'RFR', 'MLP'],
        'Negative Mean Squared Error': [svmscore,dtrscore, rfrscore, mlpscore],
        'Mean Squared Error': [svmmse, dtrmse, rfrmse, mlpmse],
        'R^2 Score': [svmr2, dtrr2, rfrr2, mlpr2],
        'Parameters': [svmparams, dtrparams, rfrparams, mlpparams]
        })

    # Save results as csv file
    evaltable.to_csv('modeleval-'+filename+'.csv', index=False)

# Model evaluation on full data
modelevalgs(X,y,'full')

# Model evaluation on imputed data
modelevalgs(Ximpute,y,'impute')
