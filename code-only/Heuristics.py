#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import openpyxl
import os
import tslearn

# Read in the dataset
data = pd.read_csv(os.path.join("data", "Final for clustering.csv"))
kmpl_data = pd.read_csv(os.path.join("data", "Final KMPL dataset.csv"))


# In[2]:


# Get unique DEPARTMENT values
data['DEPARTMENT'].unique()


# In[3]:


data.columns


# In[4]:


# Calculate the average transaction amount for each vehicle category
data['Average_Category_Amount'] = data.groupby(['RATE CARD CATEGORY', 'District', 'Month Name'])['Transaction Amount'].transform('mean')


# In[5]:


# Flag transaction amounts that are large for a category
data['Transaction_Amount_Flag'] = data['Transaction Amount'] > data['Average_Category_Amount'] * 1.5


# In[6]:


# Check the value counts of the flag
data['Transaction_Amount_Flag'].value_counts()


# # Flag tranactions where the days between transactions are less than 2

# In[7]:


# Convert 'Transaction Date' to datetime
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])

# Sort data by 'REG_NUM' and 'Transaction Date'
data.sort_values(by=['REG_NUM', 'Transaction Date'], inplace=True)

# Calculate the difference in days between transactions for each vehicle
data['Days_Between_Transactions'] = data.groupby('REG_NUM')['Transaction Date'].diff().dt.days

# Flag transactions that occur too frequently (less than 2 days apart) and the transaction amount is greater than the average transaction amount for that vehicle category
data['Transaction_Frequency_Flag'] = (data['Days_Between_Transactions'] < 2) & (data['Transaction Amount'] > data['Average_Category_Amount'])


# In[8]:


# Check the value counts of the flag
data['Transaction_Frequency_Flag'].value_counts()


# In[9]:


# Display 10 random rows where the flag is True
data[data['Transaction_Frequency_Flag']].sample(10)


# In[10]:


data.columns


# In[11]:


# Function to calculate if the difference exceeds the threshold for each transaction
diesel_actual = [22.75, 23.34, 23.43] # Actual diesel price
gov_price = 20.64 # Government price
mean_diesel = sum(diesel_actual) / 3 # Mean diesel price
diff = mean_diesel - gov_price # Difference between mean diesel price and government price

# Create a new column called Coastal Diesel Adjusted for the difference
data['Coastal Diesel Adjusted'] = data['Coastal Diesel'] + diff

# Create a new column called price difference. If the Fuel Type is Diesel, the price difference is the difference between the Coastal Diesel Adjusted and the Government Price. If the Fuel Type is Petrol, the price difference is the difference between the Coastal Petrol and the Government Price
data['Price Difference'] = data.apply(lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) if row['Fuel Type'] == 'Diesel' else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), axis=1)

# Create a Fuel Price Flag column that flags transactions where the price difference is greater than R1
data['Fuel_Price_Flag'] = data['Price Difference'] > 1


# In[12]:


# Check the value counts of the flag
data['Fuel_Price_Flag'].value_counts()


# In[13]:


# Create a new variable called number of flags that counts the number of flags for each transaction as an integer
data['Number_of_Flags'] = data['Transaction_Amount_Flag'].astype(int) + data['Transaction_Frequency_Flag'].astype(int) + data['Fuel_Price_Flag'].astype(int)

# Convert Number_of_Flags to a categorical variable
data['Number_of_Flags'] = data['Number_of_Flags'].astype('category')


# In[14]:


# Check the value counts of the flag
data['Number_of_Flags'].value_counts()


# In[50]:


# Save the data to a new file
data.to_csv('data/Final Transactions With Flags.csv', index=False)


# In[15]:


kmpl_threshold = 5  # Set threshold for KMPL
kmpl_data['KMPL_Flag'] = kmpl_data['KMPL'] < kmpl_threshold


# In[16]:


kmpl_data['KMPL_Flag'].value_counts()


# In[53]:


# Save the KMPL flagged data to a new file
kmpl_data.to_csv('data/2021 KMPL Flagged.csv', index=False)


# # Plots of the flag vs non-flag transactions against different features

# In[17]:


import pandas as pd
import os

# Read in the data
data = pd.read_csv(os.path.join("data", "Final Transactions With Flags.csv"))


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter

def countplot(data1, title1, filename):
    # Filter the data and calculate the sum of counts for the remaining categories
    filtered_data1 = data1.to_frame()
    
    # Shorten the names for each category
    shortened_names1 = filtered_data1.index.astype(str)
    
    # Create an 8x8 figure
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14
    
    # Plot the data for the first subplot
    bars = ax1.bar(shortened_names1, filtered_data1.iloc[:, 0])
    ax1.set_xticks(range(len(shortened_names1)))
    ax1.set_xticklabels(shortened_names1, fontsize=label_font_size)
    ax1.set_ylabel('Count', fontsize=y_label_font_size)
    ax1.set_xlabel('Number of Flags', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')
    
    # Add count values above each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 str(int(height)), ha='center', va='bottom', fontsize=12)
    
    # Set the y-axis ticks and labels to integer values for both subplots
    yticks1 = ax1.get_yticks().astype(int)
    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', bbox_inches='tight')
    
    # Close the plot
    plt.close(fig)

def boxplot_side_by_side_cont(data, cat_var, cont_var1, cont_var2, title1, title2, filename):
    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Plot the first boxplot
    sns.boxplot(x=cat_var, y=cont_var1, data=data, ax=ax1, palette="cividis")
    ax1.set_ylabel(cont_var1, fontsize=14)
    ax1.set_xlabel('Number of Flags', fontsize=14)
    ax1.set_title(f'a) {title1}')

    # Plot the second boxplot
    sns.boxplot(x=cat_var, y=cont_var2, data=data, ax=ax2, palette="cividis")
    ax2.set_ylabel(cont_var2, fontsize=14)
    ax2.set_xlabel('Number of Flags', fontsize=14)
    ax2.set_title(f'b) {title2}')

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file with high resolution
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', dpi=300)

    # Close the plot
    plt.close(fig)


# In[15]:


# Plot of flag against transaction amount
import matplotlib.pyplot as plt
import seaborn as sns

# Create a new dataframe called data2 with extreme values removed
data2 = data[data['Transaction Amount'] < 5000]
data2 = data2[data2['Transaction Amount'] > 0]

countplot(data['Number_of_Flags'].value_counts(), 'Number of Flags', 'countplot.pdf')

boxplot_side_by_side_cont(data2, 'Number_of_Flags', 
                          'Transaction Amount', 'No. of Litres', 
                     'Transaction Amount', 'Number of Litres',
                     'boxplots_trans_litres.pdf')


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def shorten_names(names, max_length=20):
    shortened_names = []
    for name in names:
        if len(name) > max_length:
            shortened_names.append(name[:max_length] + '...')
        else:
            shortened_names.append(name)
    return shortened_names

def four_stacked_plots(data, categorical_vars, cluster_var, titles, filename, max_categories=8, max_length=20, color_theme='tab10', show_proportions=False):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for i, (cat_var, title) in enumerate(zip(categorical_vars, titles)):
        # Calculate the proportions of each category in each cluster
        cluster_proportions = data.groupby([cluster_var, cat_var]).size().unstack(fill_value=0)
        cluster_proportions = cluster_proportions.div(cluster_proportions.sum(axis=1), axis=0)

        # Get the top categories and group the rest into "Others"
        top_categories = cluster_proportions.sum().nlargest(max_categories).index
        cluster_proportions["Others"] = cluster_proportions.drop(columns=top_categories).sum(axis=1)
        cluster_proportions = cluster_proportions[top_categories.tolist() + ["Others"]]

        # Shorten the category names if necessary
        shortened_names = shorten_names(cluster_proportions.columns, max_length=max_length)

        # Get the specified color theme
        color_scheme = plt.cm.get_cmap(color_theme, len(cluster_proportions.columns))
        colors = color_scheme(range(len(cluster_proportions.columns)))

        # Create the stacked bar chart in the corresponding subplot
        cluster_proportions.plot(kind='bar', stacked=True, ax=axs[i], legend=False, color=colors)
        axs[i].set_xticklabels(cluster_proportions.index, rotation=0, fontsize=12)
        axs[i].set_xlabel('Number of Flags', fontsize=14)
        axs[i].set_ylabel('Proportion', fontsize=14)
        axs[i].set_title(f"{chr(97+i)}) {title}")  # Prepend "a) ", "b) ", "c) ", "d) " to the titles

        if show_proportions:
            # Display the actual proportion numbers on the stacked bar plots if the proportion is greater than 0.1
            for j, rect in enumerate(axs[i].patches):
                height = rect.get_height()
                if height > 0.05:
                    axs[i].text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2,
                                f"{height:.2f}", ha='center', va='center', fontsize=9, color='black')

        # Create the legend for each subplot
        axs[i].legend(title='Categories', fontsize=10, labels=shortened_names, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

def create_proportions_tables(data, categorical_vars, cluster_var, titles, max_categories=8, max_length=20):
    tables = []

    for cat_var, title in zip(categorical_vars, titles):
        # Calculate the proportions of each category in each cluster
        cluster_proportions = data.groupby([cluster_var, cat_var]).size().unstack(fill_value=0)
        cluster_proportions = cluster_proportions.div(cluster_proportions.sum(axis=1), axis=0)

        # Get the top categories and group the rest into "Others"
        top_categories = cluster_proportions.sum().nlargest(max_categories).index
        cluster_proportions["Others"] = cluster_proportions.drop(columns=top_categories).sum(axis=1)
        cluster_proportions = cluster_proportions[top_categories.tolist() + ["Others"]]

        # Shorten the category names if necessary
        shortened_names = shorten_names(cluster_proportions.columns, max_length=max_length)
        cluster_proportions.columns = shortened_names

        # Multiply by 100 and round to 2 decimal places
        cluster_proportions = cluster_proportions.round(2)

        # Add the title as a column to the table
        cluster_proportions.insert(0, 'Flags', cluster_proportions.index)
        cluster_proportions.index = [title] * len(cluster_proportions)

        # Add the table to the list of tables
        tables.append(cluster_proportions)

    return tables


# In[33]:


data.columns


# In[19]:


# Change the data type of 'Number_of_Flags' to 'category' and order the categories
data['Number_of_Flags'] = data['Number_of_Flags'].astype('category')
data['Number_of_Flags'] = data['Number_of_Flags'].cat.reorder_categories([0, 1, 2, 3])
data['Number_of_Flags'] = data['Number_of_Flags'].cat.as_ordered()


# In[27]:


four_stacked_plots(data,
                   ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                   'Number_of_Flags',
                   ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                   'heuristics_categorical.pdf',
                   max_categories=5, max_length=15, show_proportions=True)


# In[28]:


create_proportions_tables(data,
                        ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                        'Number_of_Flags',
                        ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                        max_categories=5, max_length=15)

