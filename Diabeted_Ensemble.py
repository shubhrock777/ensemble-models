import pandas as pd
import numpy as np


#loading the dataset
df = pd.read_csv("D:/BLR10AM/Assi/17.Ensemble techniques/Datasets_ET/Diabeted_Ensemble.csv")


#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary


d_types =["count","ratio","ratio","ratio","ratio","ratio","ratio","ratio","binary"]

data_details =pd.DataFrame({"column name":df.columns,
                            "data types ":d_types,
                            "data types-p":df.dtypes})


            #3.	Data Pre-processing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of df 
df.info()
df.describe()          


#data types        
df.dtypes


#checking for na value
df.isna().sum()
df.isnull().sum()
df.dropna()

#checking unique value for each columns
df.nunique()

#variance of df
df.var()



"""4.	Exploratory Data Analysis (EDA):
4.1.	Summary
4.2.	Univariate analysis
4.3.	Bivariate analysis
	 """
    


EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA

# covariance for data set 
covariance = df.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])



# Normalization function using z std. all are continuous data.
def norm_func(x):
    y=(x-x.mean())/(x.std())
    return (y) 

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:8])
df_norm.describe()


# n-1 dummy variables will be created for n categories
df_dummy = pd.get_dummies(df.iloc[:,0], drop_first = True)
            
predictors_df_lb=pd.concat([df.iloc[:,0],df_norm],axis=1)
predictors_df_eoc = pd.concat([df_norm,df_dummy],axis=1)

target_df = df.iloc[:,8]



"""
5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Bagging, Boosting, Voting, Stacking on given datasets
5.3	Train and Test the data, use grid search cross validation, compare accuracies using confusion matrix
5.4	Briefly explain the model output in the documentation
 """

                                   #building bagging model 

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors_df_lb,target_df, test_size = 0.2,random_state=7)
                                         #for bagging we use label encoded data      


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators =500,
                            bootstrap = True, n_jobs = 1, random_state = 77)

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))


 


                                     #building  boosting model 



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors_df_lb,target_df, test_size = 0.2,random_state=7)
                                         #for bagging we use label encoded data     

#decision tree
dt = DecisionTreeClassifier() #storing the classifer in dt

dt.fit(x_train, y_train) #fitting te model 

dt.score(x_test, y_test) #checking the score like accuracy

dt.score(x_train, y_train)
#so our model is overfitting 

                                      # Ada boosting 
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=1)
ada.fit(x_train,y_train)

ada.score(x_test,y_test)

ada.score(x_train,y_train)
       

                                   #building  voting model 
                                   
# Splitting data into training and testing data set

x_train, x_test, y_train, y_test = train_test_split(predictors_df_eoc,target_df, test_size = 0.2,random_state=7)
                                          
from sklearn.ensemble import VotingClassifier
# Voting Classifier 
from sklearn.linear_model import LogisticRegression # importing logistc regression
from sklearn.svm import SVC # importing Svm 

lr = LogisticRegression() 
dt = DecisionTreeClassifier()
svm = SVC(kernel= 'poly', degree=2)

evc = VotingClassifier(estimators=[('lr', lr),('dt', dt),('svm', svm)], voting='hard')

evc.fit(x_train, y_train)

evc.score(x_test, y_test)

evc.score(x_train, y_train)




                                  #building stacking model  
                                  
                                  
 #Libraries and data loading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics


                                   
# Splitting data into training and testing data set

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
target_df_array= labelencoder.fit_transform(target_df)
target_df

predictors_df_array= predictors_df_eoc.to_numpy() 


train_x, test_x, train_y, test_y = train_test_split(predictors_df_array,target_df_array, test_size = 0.2,random_state=7)
                                         #for bagging we use label encoded data  

# Create the ensemble's base learners and meta learner
# Append base learners to a list
base_learners = []

knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)
base_learners.append(mlpc)


meta_learner = LogisticRegression(solver='lbfgs')


# Create the training meta data

# Create variables to store meta data and the targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
    # Train each learner on the K-1 folds and create meta data for the Kth fold
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(train_x[train_indices], train_y[train_indices])
        predictions = learner.predict_proba(train_x[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
    meta_index += len(test_indices)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(test_y, learner.predict(test_x))
    base_acc.append(acc)
test_meta_data = test_meta_data.transpose()

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')
