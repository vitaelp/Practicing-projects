import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cpi = pd.read_csv("CPI.csv")
sp500 = pd.read_csv("SP500.csv")

# Converting the string to datetime64 format
sp500["Date"] = pd.to_datetime(sp500["Date"])
cpi["Date"] = pd.to_datetime(cpi[['Year','Month','Day']])

#print(cpi.head(10))
#print(sp500.head())

#Plotting Simple Graph For both SP 500 Index & CPI
# Create simple line plot of S&P 500
#plt.plot(sp500["Date"], sp500["close"])
#plt.title("S&P 500 Overview")
#plt.xlabel("Date")
#plt.ylabel("SP 500 Index")
#plt.show()


#plt.plot(cpi["Date"], cpi["Actual"])
#plt.title("CPI Overview")
#plt.xlabel("Date")
#plt.ylabel("Customer Price Index")
#plt.show()

#Merging the two Dataframes & visualization
#combined_df = cpi.merge(sp500,how='inner',on='Date')
#combined_df['Actual_Enlarged'] = combined_df['Actual']*50000
#plt.plot(combined_df["Date"],combined_df['Actual_Enlarged'])
#plt.plot(combined_df["Date"],combined_df['close'])
#plt.show()

#Simple SP500 Index Prediction
X = sp500.iloc[:, 1:].values
y = sp500.iloc[:, -1].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test the model
y_pred = regressor.predict(X_test)
print(y_pred)

# Visualize the results
plt.scatter(X_test[:, 0], y_test, color='red')
plt.plot(X_test[:, 0], y_pred, color='blue')
plt.title('Stock Price Prediction')
plt.xlabel('Feature 1')
plt.ylabel('Price')
plt.show()