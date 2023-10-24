<h1>Business Problem</h1>
A machine learning project is desired to be conducted for predicting the prices of different types of houses using a dataset containing features of each house and their respective prices.
<h1>Data Story</h1>
An advanced regression techniques competition hosted on Kaggle is described in the dataset, consisting of residential homes in Ames, Iowa, with 79 explanatory variables. The dataset includes two separate CSV files for training and testing since it's associated with a Kaggle competition. The test dataset contains empty values for house prices, and participants are expected to predict these values. You can access the dataset and competition page using the following link: House Prices: Advanced Regression Techniques.
<h2>Project Tasks</h2>
<h3>Task 1: Exploratory Data Analysis</h3>
Step 1: Read and merge the Train and Test datasets. Continue working with the merged data.</br>
Step 2: Identify numerical and categorical variables.</br>
Step 3: Make necessary adjustments, such as fixing data type errors.</br>
Step 4: Explore the distribution of numerical and categorical variables in the data.</br>
Step 5: Examine the relationship between categorical variables and the target variable.</br>
Step 6: Investigate the presence of outliers.</br>
<h3>Task 2: Feature Engineering</h3>
Step 1: Perform necessary operations for missing and outlier observations.</br>
Step 2: Apply the Rare Encoder.</br>
Step 3: Create new variables.</br>
Step 4: Perform encoding operations.</br>
<h3>Task 3 : Model Creation</h3>
Step 1: Separate the Train and Test data (Data with missing SalePrice values is the test data).</br>
Step 2: Build a model using the Train data and evaluate its performance.</br>
Bonus: Build a model using the Train data after applying a logarithmic transformation to the target variable (SalePrice), and observe the RMSE results. Note: Don't forget to take the inverse of the logarithmic transformation.</br>
Step 3: Perform hyperparameter optimization.</br>
Step 4: Examine the variable importance levels.</br>
Bonus: Predict the missing SalePrice values in the Test data and create a dataframe suitable for submission on the Kaggle page. Upload your results to Kaggle.</br>