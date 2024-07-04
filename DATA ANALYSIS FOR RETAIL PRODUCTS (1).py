#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###  DATA ANALYSIS FOR RETAIL PRODUCTS  ###


# In[1]:


### Step 1: Loading the Data

import pandas as pd
import streamlit as st
st.title("Hello world!")  # add a title


# In[3]:


import pandas as pd
import streamlit as st

st.title("Hello world!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  st.write(dataframe)


# In[ ]:





# In[4]:


data


# In[ ]:


data.head()


# In[ ]:


### Step 2: Data Cleaning

missing_values = data.isnull().sum()
print(missing_values)


# In[ ]:


# Drop rows with missing values or fill them with appropriate values

data = data.dropna()  # or data.fillna(method='ffill') for forward filling


# In[ ]:


data.duplicated().sum()


# In[ ]:


# Convert 'Date' column to datetime type

data['Date'] = pd.to_datetime(data['Date'])


# In[ ]:


# Display cleaned data

data


# In[ ]:


data.columns


# In[ ]:


### Step 3: Exploratory Data Analysis (EDA)


# In[ ]:


# Summary statistics

summary_stats = data.describe()
print(summary_stats)


# In[ ]:


# Group by Category and calculate total sales

category_sales = data.groupby('Product Category').agg({'Units Sold': 'sum', 'Unit Price': 'mean'}).reset_index()
print(category_sales)


# In[ ]:


# Calculate total revenue

data['Revenue'] = data['Unit Price'] * data['Units Sold']
total_revenue = data['Revenue'].sum()
print(f"Total Revenue: ${total_revenue}")


# In[ ]:


# Monthly sales trend

monthly_sales = data.set_index('Date').resample('M').agg({'Units Sold': 'sum', 'Revenue': 'sum'}).reset_index()
print(monthly_sales)


# In[ ]:


### Step 4: Statistical Analysis


# In[ ]:


# Correlation between Price and Quantity Sold

correlation = data['Unit Price'].corr(data['Units Sold'])
print(f"Correlation between Price and Units Sold: {correlation}")


# In[ ]:


# Average revenue per product category

avg_revenue_per_category = data.groupby('Product Category')['Revenue'].mean()
print(avg_revenue_per_category)


# In[ ]:


### Step 5: Visualization


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Sales by category

plt.figure(figsize=(10, 6))
sns.barplot(x='Product Category', y='Units Sold', data=category_sales)
plt.title('Sales by Category')
plt.show()


# In[ ]:


# Monthly sales trend

plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Units Sold', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.show()


# In[ ]:


# Revenue distribution

plt.figure(figsize=(10, 6))
sns.histplot(data['Revenue'], bins=30, kde=True)
plt.title('Revenue Distribution')
plt.show()


# In[ ]:


### Step 6: Predictive Modeling 


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


# Fit the model

model = ExponentialSmoothing(monthly_sales['Units Sold'],trend='add', seasonal_periods=12)
fit = model.fit()


# In[ ]:


# Forecast the next 12 months

forecast = fit.forecast(12)
print(forecast)


# In[ ]:


# Plot the forecast

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['Date'], monthly_sales['Units Sold'], label='Actual')
plt.plot(monthly_sales['Date'].iloc[-1] + pd.to_timedelta(range(1, 13)), forecast, label='Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()

