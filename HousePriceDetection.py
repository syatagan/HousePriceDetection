#########################################################
# Business Problem
# Using the dataset of the features and house prices of each house,
# a machine learning project on the prices of different types of houses
# is desired to be carried out
#########################################################
#
# Dataset Story
# This dataset of residential homes in Lowa, Ames contains 79 explanatory variables. A quiz on Kaggle
# You can reach the dataset and competition page of the project in the link below. The dataset belongs to a kaggle competition
# There are two different csv files, train and test. House prices were left blank in the test data set, and this
# You are expected to estimate the values

# imports
from Src.utils import grab_col_names,replace_with_thresholds, check_outlier, check_MissingValue
from Src.utils import check_df,cat_summary,num_summary,target_summary_with_num,target_summary_with_cat
from Src.utils import plot_importance, outlier_analyser, missing_value_analyser, one_hot_encoder
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score, confusion_matrix , mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier

# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.max_rows",None)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

##################################################################
# DUTY 1 : Explonatary Data Analysis
##################################################################
# Stage 1 : Read and combine the Train and Test datasets. Proceed through the data you have combined

df_train = pd.read_csv("Datasets/train.csv")
df_test = pd.read_csv("Datasets/test.csv")

train_ID = df_train['Id']
test_ID = df_test['Id']
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)

df = pd.concat([df_train, df_test] )
df_train.shape
df_test.shape
df.shape
df.columns = [col.lower() for col in df.columns]
df_train.columns = [col.lower() for col in df_train.columns]
df_test.columns = [col.lower() for col in df_test.columns]
xsamplenumber = df.shape[0]
df_train.head()
# deleting outliers
df_train = df_train.drop(df_train[(df_train['grlivarea']>4000) & (df_train['saleprice']<300000)].index)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
df_train["saleprice"] = np.log1p(df_train["saleprice"])
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.saleprice.values


df = pd.concat((df_train, df_test)).reset_index(drop=True)
df.drop(['saleprice'], axis=1, inplace=True)
df.head()



check_df(df,xplot=False)
df.dtypes

# Stage 2 : Grab numeric and categorigal variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.append("neighborhood")
# Stage 3 : Make the necessary arrangements.

# Stage 4 : Observe the distribution of numerical and categorical variables in the data
for col in cat_cols:
    cat_summary(df, col, True)
for col in num_cols:
    num_summary(df, col, True)
    sns.boxplot(df[col])
    plt.show(block=True)
df.head()
# Stage 5 : Do the target variable analysis with categorical variables
for col in cat_cols:
    target_summary_with_cat(df, "saleprice", col )

for col in num_cols:
    target_summary_with_num(df, "saleprice", col )

# Stage 6 : Check for outlier values.
outlier_analyser(df , num_cols)
for col in num_cols:
    replace_with_thresholds(df,col)

# Stage 7 : Check for Missing Values , "
missing_value_analyser(df, df.columns)
df["poolqc"] = df["poolqc"].fillna("None")
df["miscfeature"] = df["miscfeature"].fillna("None")
df['alley'] = df['alley'].fillna("None")
df['fence'] = df['fence'].fillna("None")
df['fireplacequ'] = df['fireplacequ'].fillna("None")
df["lotfrontage"] = df.groupby("neighborhood")["lotfrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('garagetype', 'garagefinish', 'garagequal', 'garagecond'):
    df[col] = df[col].fillna('None')

for col in ('garageyrblt', 'garagearea', 'garagecars'):
    df[col] = df[col].fillna(0)

for col in ('bsmtfinsf1', 'bsmtfinsf2', 'bsmtunfsf','totalbsmtsf', 'bsmtfullbath', 'bsmthalfbath'):
    df[col] = df[col].fillna(0)

for col in ('bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2'):
    df[col] = df[col].fillna('None')

df["masvnrtype"] = df["masvnrtype"].fillna("None")
df["masvnrarea"] = df["masvnrarea"].fillna(0)

df['mszoning'] = df['mszoning'].fillna(df['mszoning'].mode()[0])
df = df.drop(['utilities'], axis=1)
df["functional"] = df["functional"].fillna("Typ")
df['electrical'] = df['electrical'].fillna(df['electrical'].mode()[0])
df['kitchenqual'] = df['kitchenqual'].fillna(df['kitchenqual'].mode()[0])
df['exterior1st'] = df['exterior1st'].fillna(df['exterior1st'].mode()[0])
df['exterior2nd'] = df['exterior2nd'].fillna(df['exterior2nd'].mode()[0])
df['saletype'] = df['saletype'].fillna(df['saletype'].mode()[0])
df['mssubclass'] = df['mssubclass'].fillna("None")

df['mssubclass'] = df['mssubclass'].apply(str)


#Changing OverallCond into a categorical variable
df['overallcond'] = df['overallcond'].astype(str)


#Year and month sold are transformed into categorical features.
df['yrsold'] = df['yrsold'].astype(str)
df['mosold'] = df['mosold'].astype(str)

cols = ('fireplacequ', 'bsmtqual', 'bsmtcond', 'garagequal', 'garagecond',
        'exterqual', 'extercond','heatingqc', 'poolqc', 'kitchenqual', 'bsmtfintype1',
        'bsmtfintype2', 'functional', 'fence', 'bsmtexposure', 'garagefinish', 'landslope',
        'lotshape', 'paveddrive', 'street', 'alley', 'centralair', 'mssubclass', 'overallcond',
        'yrsold', 'mosold')

df['totalsf'] = df['totalbsmtsf'] + df['1stflrsf'] + df['2ndflrsf']
from scipy import stats
numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = df[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #df[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)






################################################################
# DUTY 2 : Feature Engineering
#################################################################
# Stage 1 : Take necessary actions for missing and outlier values.
#################################################################
# # ENCODING
#################################################################
# delete id
df.drop("id", axis=1, inplace=True)
df.head()

# Stage 2 : Create new variables

# Stage 3 : run encoding for categorical variables.
cat_cols.remove("utilities")
ohe_cols = [col for col in cat_cols if  col not in cols]
for col in ohe_cols:
    df = one_hot_encoder(df,ohe_cols, drop_first=True)

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(df[c].values))
    df[c] = lbl.transform(list(df[c].values))


df.dtypes
df.shape
df_train = df[:ntrain]
df_test = df[ntrain:]
df_train.shape
df_test.shape

num_cols = [col for col in df.columns if df[col].dtypes == 'float64']
###################################################################
# STANDARDIZATION
###################################################################
# Stage 4 : Make Standardization for num_cols
df.columns = df.columns.str.replace(' ', '')
##df.columns[df.columns.str.contains(' ')]
rs = RobustScaler()
scale_cols = [col for col in num_cols if 'saleprice' not in col]
for col in num_cols:
    df[col] = rs.fit_transform(df[[col]])
df.describe().T

################################################################
# DUTY 2 : Crete Models
#################################################################

#######################################################################
## SET MODEL
#######################################################################
df.shape
df_test.shape
df_train.shape


df_train.shape
df_test.shape

df_test = df[df["saleprice"].isna()]
df_train = df[~(df["saleprice"].isna())]

X= df_train
y= y_train
X_test = df_test

reg_model = LinearRegression().fit(X,y)
y_pred = reg_model.predict(X_test)

lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["neg_mean_squared_error"])

y_pred_final = lgbm_model.predict(X_test)

print(cv_results["test_neg_mean_squared_error"].mean())




