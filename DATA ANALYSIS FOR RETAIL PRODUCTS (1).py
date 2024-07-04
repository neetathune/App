#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###  DATA ANALYSIS FOR RETAIL PRODUCTS  ###


# In[1]:


### Step 1: Loading the Data

import pandas as pd


# In[2]:


data = pd.read_csv("Online Sales Data.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


### Step 2: Data Cleaning

missing_values = data.isnull().sum()
print(missing_values)


# In[6]:


# Drop rows with missing values or fill them with appropriate values

data = data.dropna()  # or data.fillna(method='ffill') for forward filling


# In[7]:


data.duplicated().sum()


# In[8]:


# Convert 'Date' column to datetime type

data['Date'] = pd.to_datetime(data['Date'])


# In[9]:


# Display cleaned data

data


# In[10]:


data.columns


# In[11]:


### Step 3: Exploratory Data Analysis (EDA)


# In[12]:


# Summary statistics

summary_stats = data.describe()
print(summary_stats)


# In[13]:


# Group by Category and calculate total sales

category_sales = data.groupby('Product Category').agg({'Units Sold': 'sum', 'Unit Price': 'mean'}).reset_index()
print(category_sales)


# In[14]:


# Calculate total revenue

data['Revenue'] = data['Unit Price'] * data['Units Sold']
total_revenue = data['Revenue'].sum()
print(f"Total Revenue: ${total_revenue}")


# In[15]:


# Monthly sales trend

monthly_sales = data.set_index('Date').resample('M').agg({'Units Sold': 'sum', 'Revenue': 'sum'}).reset_index()
print(monthly_sales)


# In[16]:


### Step 4: Statistical Analysis


# In[17]:


# Correlation between Price and Quantity Sold

correlation = data['Unit Price'].corr(data['Units Sold'])
print(f"Correlation between Price and Units Sold: {correlation}")


# In[18]:


# Average revenue per product category

avg_revenue_per_category = data.groupby('Product Category')['Revenue'].mean()
print(avg_revenue_per_category)


# In[19]:


### Step 5: Visualization


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


# Sales by category

plt.figure(figsize=(10, 6))
sns.barplot(x='Product Category', y='Units Sold', data=category_sales)
plt.title('Sales by Category')
plt.show()


# In[22]:


# Monthly sales trend

plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Units Sold', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.show()


# In[23]:


# Revenue distribution

plt.figure(figsize=(10, 6))
sns.histplot(data['Revenue'], bins=30, kde=True)
plt.title('Revenue Distribution')
plt.show()


# In[60]:


### Step 6: Predictive Modeling 


# In[46]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[56]:


# Fit the model

model = ExponentialSmoothing(monthly_sales['Units Sold'],trend='add', seasonal_periods=12)
fit = model.fit()


# In[57]:


# Forecast the next 12 months

forecast = fit.forecast(12)
print(forecast)


# In[63]:


# Plot the forecast

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['Date'], monthly_sales['Units Sold'], label='Actual')
plt.plot(monthly_sales['Date'].iloc[-1] + pd.to_timedelta(range(1, 13)), forecast, label='Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()

