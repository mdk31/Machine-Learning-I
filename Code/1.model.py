import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#-- Read in model evaluation data --
modelimpute = pd.read_csv('modeleval-impute.csv')
modelimpute['Data'] = "impute"
modelfull = pd.read_csv('modeleval-full.csv')
modelfull['Data'] = "full"

model = pd.concat([modelimpute,modelfull])

#rank models by lowest MSE
model['Rank'] = model['Mean Squared Error'].rank(ascending=True)

best_model = model.loc[model['Rank']==1].reset_index()
#hold parameters from highest ranked model
paramsdict = best_model['Parameters'].to_dict()
params = eval(paramsdict[0])

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
print("Shape:", df.shape)

# Check first 10 rows
print("Head:\n", df.head(10))
# Check info of dataset

print("Statistics of dataset:\n", df.info(verbose=True))
# replace ? values with numpy nan
df = df.replace('?',np.nan)

# Find the number of columns with missing values
print("Number of columns with nulls:", df.isnull().any().sum(axis=0))

missing = df.isnull().sum()/(len(df))*100
#check columns with over 50% missing values
print("Over 50% Null:\n", missing.where(missing>50))
#check for total null values in df
print("Total Nulls:",df.isnull().sum().sum())

#drop non predictive fields
df = df.drop(columns =['state','county','communityname','community','fold',])

#set X and y
X = df.drop('ViolentCrimesPerPop', axis=1)
y = df['ViolentCrimesPerPop']
columns = X.columns

#impute X values
if best_model.iloc[0][6] == 'impute':
    imp = IterativeImputer(max_iter=10, random_state=0)
    X = imp.fit_transform(X)
    X = pd.DataFrame(X, columns = columns)

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
    return(newOrder)



#remove columns with null values
X = X.dropna(axis=1)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#remove highly correlated features
to_remove = findCorrelations(X_train.corr())
X_train = X_train.drop(X_train.columns[to_remove], axis=1)
X_test = X_test.drop(X_test.columns[to_remove], axis=1)

print("X_train shape:", X_train.shape, "\nX_test shape:", X_test.shape)

#-- Model --
if best_model.iloc[0][1] == 'MLP':
    mlp = MLPRegressor(hidden_layer_sizes=params['mlp__hidden_layer_sizes'], activation=params['mlp__activation'])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

elif best_model.iloc[0][1] == 'RFR':
    rfr = RandomForestRegressor(bootstrap=params['bootstrap'], max_depth=params['max_depth'], max_features='sqrt', n_estimators=params['n_estimators'])
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

elif best_model.iloc[0][1] == 'DTR':
    dtr = DecisionTreeRegressor(max_depth=params['max_depth'],max_features=params['max_features'])
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)

else :
    svmr = SVR(C=params['C'], epsilon=params['epsilon'], kernel=params['kernel'])
    svmr.fit(X_train, y_train)
    y_pred = svmr.predict(X_test)

#-- Plot Comparison --
plt.figure(0)
plt.title('Violent Crimes per Population Comparison')
plt.hist(y_test,bins=20,label='Actual',alpha=0.5)
plt.hist(y_pred, bins=20, label='Predicted',alpha=0.5)
plt.legend()
plt.savefig('Histogram_Comparison', dpi=300, bbox_inches='tight')



def featuretest(feature):
    """
    Test and plot change in predicted values by change in police features

    :param feature: name of feature to test

    :return: scatter plot of change in values
    """

    #Create test value of mean values for each feature
    meantest = X.mean().to_frame().transpose()
    meantest = meantest.drop(meantest.columns[to_remove], axis=1)
    feature = str(feature)

    #empty array to hold predicted values
    predictions = []
    #each new value to test
    val = np.arange(0,1.01,0.05)

    for i in val:
        meantest[feature] = i
        pred = mlp.predict(meantest)
        predictions.append(pred)

    # create scatter plot of change in feature and prediction value
    plt.figure(1).clear(True)
    plt.figure(1)
    plt.title("Change in Predicted Violent Crimes per Population by Change in "+feature)
    plt.xlabel(feature)
    plt.ylabel('Predicted Violent Crimes per Population')
    plt.scatter(val,predictions)
    plt.savefig("Scatter-"+feature, dpi=300, bbox_inches='tight')

if best_model.iloc[0][6] == 'impute':
    featuretest('PolicPerPop')
    featuretest('PolicOperBudg')

else:
    featuretest('PctUnemployed')
    featuretest('PctEmploy')
    featuretest('PopDens')
    featuretest('LandArea')






