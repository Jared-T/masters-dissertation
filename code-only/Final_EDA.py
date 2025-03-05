#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# Read in the dataset
trans_df = pd.read_csv(os.path.join("data", "Final with Coords, Fuel Type and Prices.csv"))
vehicle_df = pd.read_csv(os.path.join("data", "Final KMPL dataset.csv"))


# In[2]:


trans_df.shape


# In[3]:


vehicle_df.columns


# In[4]:


trans_df.columns


# In[5]:


# Remove the word "CATEGORY" from the 'RATE CARD CATEGORY' in vehicle_df
vehicle_df['RATE CARD CATEGORY'] = vehicle_df['RATE CARD CATEGORY'].str.replace('CATEGORY', '')

trans_df['RATE CARD CATEGORY'] = trans_df['RATE CARD CATEGORY'].str.replace('CATEGORY', '')


# In[6]:


# Calculate the number of days between transactions for each vehicle
trans_df['Transaction Date'] = pd.to_datetime(trans_df['Transaction Date'])

# Sort data by 'REG_NUM' and 'Transaction Date'
trans_df.sort_values(by=['REG_NUM', 'Transaction Date'], inplace=True)

# Calculate the difference in days between transactions for each vehicle
trans_df['Days Between Transactions'] = trans_df.groupby('REG_NUM')['Transaction Date'].diff().dt.days


# In[17]:


trans_df['Days Between Transactions'].isnull().sum()


# In[20]:


trans_df['REG_NUM'].unique().shape


# In[81]:


trans_df.columns


# In[7]:


import random
# Select a random vehicle to check the days between transactions
# Select a random vehicle
random_vehicle = random.choice(trans_df['REG_NUM'].unique())

# Filter the DataFrame for the selected vehicle
selected_vehicle = trans_df[trans_df['REG_NUM'] == random_vehicle]

# Check the days between transactions for the selected vehicle
days_between_transactions = selected_vehicle[['REG_NUM', 'Transaction Date', 'MODEL DERIVATIVE',
                                              'No. of Litres', 'Days Between Transactions']]

# Print the result
days_between_transactions



# In[8]:


# Get the average number of days between transactions for each vehicle and add it to the vehicle_df
average_days_between_transactions = trans_df.groupby('REG_NUM')['Days Between Transactions'].mean()

average_days_between_transactions.head()


# In[9]:


# Merge the average_days_between_transactions with the vehicle_df
vehicle_df = vehicle_df.merge(average_days_between_transactions, how='left', left_on='REG_NUM', right_on='REG_NUM')

# Rename the column
vehicle_df.rename(columns={'Days Between Transactions': 'Average Days Between Transactions'}, inplace=True)


# # Fleet Composition

# In[10]:


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns

def shorten_names(names, max_length=20):
    shortened_names = []
    for name in names:
        if len(name) > max_length:
            shortened_name = name[:max_length-3] + '...'
        else:
            shortened_name = name
        shortened_names.append(shortened_name)
    return shortened_names


def create_countplot(data, filename, threshold, max_length=50):
    # Filter the data and calculate the sum of counts for the remaining categories
    filtered_data = data[data >= threshold]
    others_count = data[data < threshold].sum()
    filtered_data['Others'] = others_count

    # Shorten the names for each category
    shortened_names = shorten_names(filtered_data.index, max_length=max_length)

    # Create a single plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set the font size for the title and labels
    label_font_size = 12
    y_label_font_size = 14

    # Plot the data
    ax.bar(shortened_names, filtered_data)
    ax.set_xticklabels(shortened_names, rotation=45, ha='right', fontsize=label_font_size)
    ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=label_font_size)
    ax.set_ylabel('Count', fontsize=y_label_font_size)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'plots/eda/{filename}', format='pdf', bbox_inches='tight')

    # Close the plot
    plt.close(fig)


def countplot_side_by_side(data1, data2, title1, title2, filename, threshold1, threshold2, max_length=50):
    # Filter the data and calculate the sum of counts for the remaining categories
    filtered_data1 = data1[data1 >= threshold1]
    others_count1 = data1[data1 < threshold1].sum()
    filtered_data1 = filtered_data1.to_frame()

    if others_count1 > 0:
        filtered_data1.loc['Others'] = others_count1

    filtered_data2 = data2[data2 >= threshold2]
    others_count2 = data2[data2 < threshold2].sum()
    filtered_data2 = filtered_data2.to_frame()

    if others_count2 > 0:
        filtered_data2.loc['Others'] = others_count2

    # Shorten the names for each category
    shortened_names1 = shorten_names(filtered_data1.index, max_length=max_length)
    shortened_names2 = shorten_names(filtered_data2.index, max_length=max_length)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14

    # Plot the data for the first subplot
    ax1.bar(shortened_names1, filtered_data1.iloc[:, 0])
    ax1.set_xticklabels(shortened_names1, rotation=45, ha='right', fontsize=label_font_size)
    ax1.set_ylabel('Count', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')

    # Plot the data for the second subplot
    ax2.bar(shortened_names2, filtered_data2.iloc[:, 0])
    ax2.set_xticklabels(shortened_names2, rotation=45, ha='right', fontsize=label_font_size)
    ax2.set_title(f'b) {title2}')

    # Set the y-axis ticks and labels to integer values for both subplots
    yticks1 = ax1.get_yticks().astype(int)
    yticks2 = ax2.get_yticks().astype(int)
    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)
    ax2.yaxis.set_major_locator(FixedLocator(yticks2))
    ax2.yaxis.set_major_formatter(FixedFormatter(yticks2))
    ax2.tick_params(axis='y', labelsize=label_font_size)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'plots/eda/{filename}', format='pdf', bbox_inches='tight')

    # Close the plot
    plt.close(fig)


def histogram_side_by_side(data1, var1, data2, var2, title1, title2, filename, bins=20):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14

    # Plot the data for the first subplot
    ax1.hist(data1[var1], bins=bins, edgecolor='black')
    ax1.set_xlabel(var1, fontsize=label_font_size)
    ax1.set_ylabel('Frequency', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')

    # Plot the data for the second subplot
    ax2.hist(data2[var2], bins=bins, edgecolor='black')
    ax2.set_xlabel(var2, fontsize=label_font_size)
    ax2.set_ylabel('Frequency', fontsize=y_label_font_size)
    ax2.set_title(f'b) {title2}')

    # Set the y-axis ticks and labels to integer values for both subplots
    yticks1 = ax1.get_yticks().astype(int)
    yticks2 = ax2.get_yticks().astype(int)
    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)
    ax2.yaxis.set_major_locator(FixedLocator(yticks2))
    ax2.yaxis.set_major_formatter(FixedFormatter(yticks2))
    ax2.tick_params(axis='y', labelsize=label_font_size)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'plots/eda/{filename}', format='pdf', bbox_inches='tight')

    # Close the plot
    plt.close(fig)

def barplot_side_by_side(data, cat_var, cont_var1, cont_var2, title1, title2, filename, max_length=50, largest_n=9, show_obs_count=False):
    # Group the data by the categorical variable and calculate the mean of the continuous variables
    grouped_data1 = data.groupby(cat_var)[cont_var1].mean().nlargest(largest_n)
    grouped_data2 = data.groupby(cat_var)[cont_var2].mean().nlargest(largest_n)

    # Calculate the sum of the remaining means for the "Other" category
    other_mean1 = data[~data[cat_var].isin(grouped_data1.index)].groupby(cat_var)[cont_var1].mean().mean()
    other_mean2 = data[~data[cat_var].isin(grouped_data2.index)].groupby(cat_var)[cont_var2].mean().mean()

    # Append the "Other" category to the grouped data
    grouped_data1['Other'] = other_mean1
    grouped_data2['Other'] = other_mean2

    # Shorten the names for each category
    shortened_names = shorten_names(grouped_data1.index, max_length=max_length)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14

    # Plot the data for the first subplot
    bars1 = ax1.bar(shortened_names, grouped_data1)
    ax1.set_xticklabels(shortened_names, rotation=45, ha='right', fontsize=label_font_size)
    ax1.set_ylabel(f'Average {cont_var1}', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')

    # Add the number of observations on top of each bar for the first subplot if show_obs_count is True
    if show_obs_count:
        for i, name in enumerate(shortened_names):
            obs_count = data[data[cat_var] == grouped_data1.index[i]].shape[0]
            ax1.text(i, grouped_data1[i], f'{obs_count}', ha='center', va='bottom', fontsize=11)

    # Plot the data for the second subplot
    bars2 = ax2.bar(shortened_names, grouped_data2)
    ax2.set_xticklabels(shortened_names, rotation=45, ha='right', fontsize=label_font_size)
    ax2.set_ylabel(f'Average {cont_var2}', fontsize=y_label_font_size)
    ax2.set_title(f'b) {title2}')

    # Add the number of observations on top of each bar for the second subplot if show_obs_count is True
    if show_obs_count:
        for i, name in enumerate(shortened_names):
            obs_count = data[data[cat_var] == grouped_data2.index[i]].shape[0]
            ax2.text(i, grouped_data2[i], f'{obs_count}', ha='center', va='bottom', fontsize=11)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'plots/eda/{filename}', format='pdf', bbox_inches='tight')

    # Close the plot
    plt.close(fig)

def histogram_2x2(data1, var1, data2, var2, data3, var3, title1, title2, title3, filename, bins=20):
    # Create a figure with three subplots side by side
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14

    # Plot the data for the first subplot
    ax1.hist(data1[var1], bins=bins, edgecolor='black')
    ax1.set_xlabel(var1, fontsize=label_font_size)
    ax1.set_ylabel('Frequency', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')

    # Plot the data for the second subplot
    ax2.hist(data2[var2], bins=bins, edgecolor='black')
    ax2.set_xlabel(var2, fontsize=label_font_size)
    ax2.set_ylabel('Frequency', fontsize=y_label_font_size)
    ax2.set_title(f'b) {title2}')

    # Plot the data for the third subplot
    ax3.hist(data3[var3], bins=bins, edgecolor='black')
    ax3.set_xlabel(var3, fontsize=label_font_size)
    ax3.set_ylabel('Frequency', fontsize=y_label_font_size)
    ax3.set_title(f'c) {title3}')

    # Remove the fourth subplot (unused)
    ax4.axis('off')

    # Set the y-axis ticks and labels to integer values for all subplots
    yticks1 = ax1.get_yticks().astype(int)
    yticks2 = ax2.get_yticks().astype(int)
    yticks3 = ax3.get_yticks().astype(int)

    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)

    ax2.yaxis.set_major_locator(FixedLocator(yticks2))
    ax2.yaxis.set_major_formatter(FixedFormatter(yticks2))
    ax2.tick_params(axis='y', labelsize=label_font_size)

    ax3.yaxis.set_major_locator(FixedLocator(yticks3))
    ax3.yaxis.set_major_formatter(FixedFormatter(yticks3))
    ax3.tick_params(axis='y', labelsize=label_font_size)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(f'plots/eda/{filename}', format='pdf', bbox_inches='tight')

    # Close the plot
    plt.close(fig)

def boxplot_side_by_side_cat(data, cat_var1, cat_var2, cont_var, title1, title2, filename):
    # Create new columns for the top 5 categories and "Other"
    top5_cat1 = data[cat_var1].value_counts().head(5).index
    data[f'{cat_var1}_top5'] = data[cat_var1].apply(lambda x: x if x in top5_cat1 else 'Other')
    
    top5_cat2 = data[cat_var2].value_counts().head(5).index
    data[f'{cat_var2}_top5'] = data[cat_var2].apply(lambda x: x if x in top5_cat2 else 'Other')
    
    # Shorten the category names
    data[f'{cat_var1}_top5'] = shorten_names(data[f'{cat_var1}_top5'])
    data[f'{cat_var2}_top5'] = shorten_names(data[f'{cat_var2}_top5'])
    
    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Plot the first boxplot
    sns.boxplot(x=f'{cat_var1}_top5', y=cont_var, data=data, ax=ax1, palette="cividis")
    ax1.set_ylabel(cont_var, fontsize=14)
    ax1.set_xlabel("", fontsize=14)
    ax1.set_title(f'a) {title1}')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot the second boxplot
    sns.boxplot(x=f'{cat_var2}_top5', y=cont_var, data=data, ax=ax2, palette="cividis")
    ax2.set_ylabel("", fontsize=14)
    ax2.set_xlabel("", fontsize=14)
    ax2.set_title(f'b) {title2}')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file with high resolution
    plt.savefig(f'plots/eda/{filename}', format='pdf', dpi=300)
    
    # Close the plot
    plt.close(fig)


# In[11]:


# Group the DataFrame by 'MODEL DERIVATIVE' and count the occurrences
fleet_data_derivative = vehicle_df['MODEL DERIVATIVE'].value_counts()
trans_data_derivative = trans_df['MODEL DERIVATIVE'].value_counts()

# Group the DataFrame by 'VEHICLE MAKE' and count the occurrences
fleet_data_make = vehicle_df['VEHICLE MAKE'].value_counts()
trans_data_make = trans_df['VEHICLE MAKE'].value_counts()

# Group the DataFrame by 'DEPARTMENT' and count the occurrences
fleet_data_department = vehicle_df['DEPARTMENT'].value_counts()
trans_data_department = trans_df['DEPARTMENT'].value_counts()

# Group the DataFrame by 'RATE CARD CATEGORY' and count the occurrences
fleet_data_rate_card = vehicle_df['RATE CARD CATEGORY'].value_counts()
trans_data_rate_card = trans_df['RATE CARD CATEGORY'].value_counts()


# In[13]:


# Create and save individual plots
countplot_side_by_side(fleet_data_derivative, trans_data_derivative, 
                     'Fleet', 'Transactions', 
                     'model_derivative.pdf', 
                     25, 5000,
                     20)

countplot_side_by_side(fleet_data_make, trans_data_make, 
                     'Fleet', 'Transactions', 
                     'make.pdf', 
                     20, 5000,
                     20)

countplot_side_by_side(fleet_data_department, trans_data_department, 
                     'Fleet', 'Transactions', 
                     'department.pdf', 
                     20, 5000,
                     20)

countplot_side_by_side(fleet_data_rate_card, trans_data_rate_card, 
                     'Fleet', 'Transactions', 
                     'rate_card.pdf', 
                     20, 5000,
                     20)


# In[12]:


# Convert the date column to a datetime object
trans_df['Transaction Date'] = pd.to_datetime(trans_df['Transaction Date'])

# Create a new column for the month name
trans_df['Month Name'] = trans_df['Transaction Date'].dt.month_name()

# Create a new column for the weekday name
trans_df['Weekday Name'] = trans_df['Transaction Date'].dt.day_name()


# In[14]:


trans_df.columns


# In[13]:


trans_data_wday = trans_df['Weekday Name'].value_counts()
trans_data_month = trans_df['Month Name'].value_counts()


# In[16]:


# Create and save individual plots
countplot_side_by_side(trans_data_wday, trans_data_month, 
                     'Weekday', 'Month', 
                     'month_weekday.pdf', 
                     5000, 5000,
                     30)


# In[48]:


trans_data_merchant = trans_df['Merchant Name'].value_counts()
trans_data_fuel_type = trans_df['Fuel Type'].value_counts()

countplot_side_by_side(trans_data_merchant, trans_data_fuel_type, 
                     'Merchant Name', 'Fuel Type', 
                     'merchants_fuel_type.pdf', 
                     3000, 2000,
                     30)


# In[14]:


# Rename the "UNKNOWN" category to "Unknown" in the 'District' column
vehicle_df['District'] = vehicle_df['District'].replace('UNKNOWN', 'Unknown')
vehicle_df['Site'] = vehicle_df['Site'].replace('Pe', 'PE')

# Remove the values where Distict is "Unknown"
vehicle_df_nd = vehicle_df[vehicle_df['District'] != "Unknown"]
vehicle_df_ns = vehicle_df[vehicle_df['Site'] != "Unknown"]

vehicle_data_district = vehicle_df_nd['District'].value_counts()
vehicle_data_site = vehicle_df_ns['Site'].value_counts()


# In[20]:


# Create and save individual plots
countplot_side_by_side(vehicle_data_district, vehicle_data_site, 
                     'District', 'Site', 
                     'district_site.pdf', 
                     20, 8,
                     30)


# In[53]:


vehicle_df.columns


# In[15]:


# Filter out all transaction amounts that are greater than 5000
trans_df_ne = trans_df[trans_df['Transaction Amount'] <= 5000]
trans_df_ne = trans_df_ne[trans_df_ne['Transaction Amount'] > 0]

trans_df_ne = trans_df_ne[trans_df_ne['No. of Litres'] <= 200]

# Rename 'KMPL' to 'Kilometres per Litre'
vehicle_df.rename(columns={'KMPL': 'Kilometres per Litre'}, inplace=True)

histogram_2x2(trans_df_ne, 'Transaction Amount', trans_df_ne, 'No. of Litres', vehicle_df, 'Kilometres per Litre',
                       'Transaction Amount', 'Number of Litres', 'Kilometres per Litre',
                     'transaction_litres_kmpl.pdf', 
                     20)


# In[23]:


histogram_side_by_side(trans_df_ne, 'Days Between Transactions', vehicle_df, 'Average Days Between Transactions',
                       'Transactions', 'Vehicles', 
                     'days_between_transactions.pdf', 
                     20)


# ## Transaction and fleet amount means across different vars

# In[14]:


trans_df.columns


# In[58]:


barplot_side_by_side(trans_df, 'VEHICLE MAKE', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_make.pdf', 
                     25)

barplot_side_by_side(trans_df, 'MODEL DERIVATIVE', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_derivative.pdf', 
                     25)

barplot_side_by_side(trans_df, 'DEPARTMENT', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_department.pdf', 
                     25)

barplot_side_by_side(trans_df, 'RATE CARD CATEGORY', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_category.pdf', 
                     25)

barplot_side_by_side(trans_df, 'District', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_district.pdf', 
                     25)

barplot_side_by_side(trans_df, 'Month Name', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_month.pdf', 
                     25)

barplot_side_by_side(trans_df, 'Weekday Name', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_wday.pdf', 
                     25)


# In[16]:


# Define the bins for the different bands
bins = [0, 2, 5, 10, 20, 30, float('inf')]

# Define the labels for the different bands
labels = ['< 2 days', '2-5 days', '5-10 days', '10-20 days', '20-30 days', '> 30 days']

# Create the categorical variable using pd.cut()
trans_df['Days Between Transactions Category'] = pd.cut(trans_df['Days Between Transactions'], bins=bins, labels=labels)

barplot_side_by_side(trans_df, 'Days Between Transactions Category', 'Transaction Amount', 'No. of Litres',
                     'Transaction Amount', 'Number of Litres', 
                     'biplot_numdays.pdf', 
                     25, show_obs_count=True)


# In[31]:


# Calculate the average of each pair of number of litres for each vehicle
trans_df['Avg Litres'] = trans_df.groupby('REG_NUM')['No. of Litres'].rolling(window=2).mean().reset_index(0, drop=True)

# Calculate the difference in consecutive litres for each vehicle
trans_df['Litres Diff'] = trans_df.groupby('REG_NUM')['No. of Litres'].diff()

# Create a new DataFrame with the required columns
plot_df = trans_df[['Days Between Transactions Category', 'Avg Litres', 'Litres Diff']].dropna()

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Plot the first boxplot
sns.boxplot(x='Days Between Transactions Category', y='Avg Litres', data=plot_df, ax=ax1, palette="cividis")
ax1.set_ylabel('2-Day Rolling Average Litres', fontsize=14)
ax1.set_xlabel('', fontsize=14)
ax1.set_title('a) Distribution of Average Litres')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Plot the second boxplot
sns.boxplot(x='Days Between Transactions Category', y='Litres Diff', data=plot_df, ax=ax2, palette="cividis")
ax2.set_ylabel('Difference in Consecutive Litres', fontsize=14)
ax2.set_xlabel('', fontsize=14)
ax2.set_title('b) Distribution of Difference in Consecutive Litres')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Adjust the spacing
plt.tight_layout()

# Save the plot as a PDF file with high resolution
plt.savefig('plots/eda/boxplot_avgdays_avglitres_diff.pdf', format='pdf', dpi=300)

# Close the plot
plt.close(fig)


# In[38]:


# Filter the DataFrame to get the rows where 'Days Between Transactions Category' is '> 30 days'
greater_than_30_days_df = trans_df[trans_df['REG_NUM'] == 'GGA023EC']

# Sort the DataFrame by 'REG_NUM' and 'Transaction Date'
greater_than_30_days_df.sort_values(by=['REG_NUM', 'Transaction Date'], inplace=True)

# Select the 'Litres Diff' column and display the first 10 values
greater_than_30_days_df[['Transaction Date', 'REG_NUM', 'Transaction Amount', 'No. of Litres', 
                         'Days Between Transactions Category', 'Avg Litres', 'Litres Diff']].head(10)


# In[59]:


# Save the filtered dataset to a new CSV file
trans_df.to_csv(os.path.join("data", "Final for clustering.csv"), index=False)


# In[21]:


# List the first five model derivatives for vehicles that contain "12" in rate card category
vehicle_df[vehicle_df['RATE CARD CATEGORY'].str.contains('12')]['MODEL DERIVATIVE'].head()


# In[34]:


trans_df.columns


# In[35]:


# Save the 'REG_NUM', 'Merchant Lat' and 'Merchant Long' to a new CSV file
trans_df[['REG_NUM', 'Merchant Lat', 'Merchant Long']].to_csv(os.path.join("data", "Final for QGIS.csv"), index=False)


# In[36]:


trans_df[['REG_NUM', 'Merchant Lat', 'Merchant Long']]


# In[61]:


vehicle_df.columns


# In[77]:


boxplot_side_by_side_cat(vehicle_df, 'MODEL DERIVATIVE', 'RATE CARD CATEGORY', 'Kilometres per Litre',
                         'model derivative', 'rate card category', 
                         'biplot_kmpl.pdf')


# In[78]:


vehicle_df.shape


# In[ ]:




