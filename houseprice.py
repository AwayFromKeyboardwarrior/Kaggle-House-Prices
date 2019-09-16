import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

sns.set()

train = pd.read_csv('houseprice/train.csv')
test = pd.read_csv('houseprice/test.csv')

#print(train)
#print(train)
#print(train.shape)
#print(train.info())

# print(train['SalePrice'].describe())
# print(train['SalePrice'].skew())
# print(train['SalePrice'].kurt())

# sns.distplot(train['SalePrice'])
# plt.show()

#train = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
#train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
#print(train)
#sns.scatterplot(x=train['SalePrice'],y=train['GrLivArea'])

#sns.scatterplot(x=train['SalePrice'],y=train['TotalBsmtSF'])
#sns.scatterplot(x=train['SalePrice'],y=train['TotRmsAbvGrd'])
#sns.scatterplot(x=train['SalePrice'],y=train['OverallQual'])
#train = pd.concat([train['SalePrice'],train['OverallQual']],axis=1)
#plt.show()

#fig,ax = plt.subplots(figsize=(10,8))
#sns.boxplot(y=train['SalePrice'],x=train['OverallQual'],ax=ax)
#plt.show()
# fig,ax = plt.subplots()
# plt.sca(ax=ax)
# plt.xticks(rotation = 45)
# sns.boxplot(x=train['YearBuilt'],y=train['SalePrice'],ax=ax)
# plt.show()

corrmatt = train.corr()
# fig,ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmatt,square=True)
# plt.show()
#
# print(corrmatt.nlargest(10,'SalePrice'))
# cols = corrmatt.nlargest(10,'SalePrice').index
# corrmatt = train[cols].corr()
# fig,ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmatt,square=True,annot=True)
# plt.show()










#
cols = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
#sns.pairplot(train[cols],height=1.5)
#plt.show()
#print(train[cols])


a = train.isnull().sum().sort_values(ascending=False)
b = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
#print(train.shape[0])
missingdata_train = pd.concat([a,b],axis=1,keys=['Total','Percent'])
#print(missingdata_train)
#print(train.shape)

train.drop(missingdata_train[missingdata_train['Total']>1].index,axis=1,inplace=True)
#print(train.shape)

train.drop(train.loc[train['Electrical'].isnull()].index,inplace=True)
#print(train.shape)



#print(test.shape)
test.drop(missingdata_train[missingdata_train['Total']>1].index,axis=1,inplace=True)
#print(test.shape)

a = test.isnull().sum().sort_values(ascending=False)
b = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
#print(test.shape[0])
missingdata_test = pd.concat([a,b],axis=1,keys=['Total','Percent'])
missingindex_test = missingdata_test[missingdata_test['Total']>0].index
#missingindex_test = ['MSZoning', 'BsmtHalfBath', 'BsmtFullBath', 'Functional', 'Utilities']
#print(missingindex_test)
#

#print(test.shape)
for i in range(0,len(missingindex_test)):
    test[missingindex_test[i]].fillna(test[missingindex_test[i]][0],inplace=True)
    #test[missingindex_test[i]].fillna(0)
    #print(test[missingindex_test[i]][0])
#print(test.shape)

#print(test.isnull().sum().sort_values(ascending=False))

#for i in range(0,len(test.loc[:,missingindex_test].columns)):
    #print(test.loc[:,missingindex_test].iloc[:,i])
    # test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True] = test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index-1]
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index-1])
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index])
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True])
    # #test.loc[:,missingindex_test].iloc[:,i].fillna(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index-1])
    # test.loc[:,missingindex_test].iloc[:,i].fillna(test.loc[:,missingindex_test].iloc[:,i].loc[[test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index-1])
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True])
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index-1])
    # print(test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i][test.loc[:,missingindex_test].iloc[:,i].isnull()==True].index])

# a = test.isnull().sum().sort_values(ascending=False)
# b = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
#print(a,b)






#print('shape : ',train.shape)

scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])
#print(scaled)

low_range = scaled[scaled[:,0].argsort()][:10]
high_range = scaled[scaled[:,0].argsort()][-10:]

#print(low_range)
#print(high_range)

#
# sns.scatterplot(x=train['SalePrice'],y=train['GrLivArea'])
# plt.show()

#print(train.index[1])
#print(train['GrLivArea'].sort_values(ascending=True))
train.drop(train.index[523],axis=0,inplace=True)
train.drop(train.index[1297],axis=0,inplace=True)
#print(train['GrLivArea'].sort_values(ascending=True))


# sns.scatterplot(x=train['SalePrice'],y=train['GrLivArea'])
# plt.show()

#sns.distplot(train['SalePrice'],fit=norm)
#fig,axis= plt.subplots()
# stats.probplot(train['SalePrice'],plot=plt)
#stats.probplot(train['GrLivArea'],plot=plt)
#stats.probplot(train['TotalBsmtSF'],plot=plt)

# plt.show()
#sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])

train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'])
#print(train.loc[train['TotalBsmtSF']!=0]['TotalBsmtSF'])


#test['SalePrice'] = np.log(test['SalePrice'])

test['GrLivArea'] = np.log(test['GrLivArea'])

# train['test1']= train.loc[train['TotalBsmtSF']!=0]['TotalBsmtSF']
# #train['test1']=0
# train.loc[train['TotalBsmtSF']==0,'test1']=0
# print(train['test1'])
#
# stats.probplot(train['test1'],plot=plt)
#fig,axis = plt.subplots(nrows=2)
#stats.probplot(train['TotalBsmtSF'],plot=plt)


train.loc[train['TotalBsmtSF']!=0,'TotalBsmtSF'] = np.log(train.loc[train['TotalBsmtSF']!=0,'TotalBsmtSF'])


test.loc[test['TotalBsmtSF']!=0,'TotalBsmtSF'] = np.log(test.loc[test['TotalBsmtSF']!=0,'TotalBsmtSF'])


# #train['test1'] = np.log(train['test1'])
# #sns.distplot(train['SalePrice'],fit=norm)
# sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm)
# #stats.probplot(train['SalePrice'],plot=plt)
# #stats.probplot(train['GrLivArea'],plot=plt)
# #stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)
#
#
#
# plt.show()
#
# sns.scatterplot(x=train[train['TotalBsmtSF']>0]['TotalBsmtSF'],y=train[train['TotalBsmtSF']>0]['SalePrice'])
# #sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])
# plt.show()

train1 = train.drop(['SalePrice','LotConfig','Neighborhood','Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Electrical'],axis=1)
train1 = pd.get_dummies(train1)
#print(train1)
test = test.drop(['LotConfig','Neighborhood','Utilities','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Electrical'],axis=1)
test = pd.get_dummies(test)

# cols1 = train1.columns
# cols2 = test.columns
# print(cols1)
# print(cols2)
#print(list[cols1]-list[cols2])

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import OneHotEncoder
model = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
# ohe = OneHotEncoder()
# ohe.transform()
# print(train)
X_train = train1

X_test = test.loc[:,test.columns != 'SalePrice']

Y_train = train['SalePrice']
k_Fold = KFold(n_splits=10,shuffle=True,random_state=0)
#score = cross_val_score(model,X_train,Y_train,cv=k_Fold)
#print(score)

model.fit(X_train,Y_train)
predict =  model.predict(X_test)
print(predict)
print(predict.shape)

# print(model)
submission = pd.read_csv('houseprice/sample_submission.csv')
submission['SalePrice']=np.exp(predict)
submission.to_csv('submission_test.csv',index=False)