#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# Data preparation plan:
# 
# 1. Load the transaction data
# 2. Merge the transaction data with the fleet register datasets
# 3. Clean the columns and remove all unnecessary transactions
# 4. Add the coordinates of the transactions based on the merchant name
# 5. Add the true fuel prices to the transactions
# 6. Create a second dataset with the total fuel consumption per vehicle based on the tracking dataset

# # 1. Load the transaction data

# In[1]:


import pandas as pd
import openpyxl
import os


# In[2]:


# Load in the transaction dataset
raw_trans = pd.read_csv(os.path.join("..", "data", "2021 Monthly Full Transactions.csv"))


# In[3]:


# Check dataset size
raw_trans.shape


# In[4]:


# Check the data structure
raw_trans.info()


# In[6]:


# Create a new dataframe with only the first 5 rows
trans_df_for_report = raw_trans.head()

# Step 1: Identify unique registration numbers in the subset
unique_reg_nums_subset = sorted(trans_df_for_report['REG_NUM'].unique())

# Step 2: Create a mapping from each unique registration number in the subset to an anonymized value
anonymized_mapping_subset = {reg_num: 'ANON_REG_NUM_' + str(i) for i, reg_num in enumerate(unique_reg_nums_subset, start=1)}

# Step 3: Apply the mapping to your subset
trans_df_for_report['REG_NUM'] = trans_df_for_report['REG_NUM'].map(anonymized_mapping_subset)

# Save the anonymized dataset to a new CSV file
trans_df_for_report.to_csv('../data/anonymized_raw_transactions.csv', index=False)


# # 2. Merge the transaction data with the fleet register datasets

# ## 2.1 Load the fleet register datasets

# In[8]:


full_register = pd.read_csv(os.path.join("..", "data", "full_fleet_register.csv"), sep=';')

colnames = ['NEW REG  NO.', 'VEHICLE MAKE', 'MODEL DERIVATIVE', 'DEPARTMENT', 'RATE CARD CATEGORY']

full_register = full_register[colnames]

full_register = full_register.rename(columns={'NEW REG  NO.': 'Reg No'})


# In[16]:


full_register.shape


# In[9]:


white = pd.read_csv(os.path.join("..", "data", "white fleet data.csv"), sep=';')

colnames = ['Reg No', 'Old Reg', 'Site', 'District']

white = white[colnames]

white = white.rename(columns={'Rental ': 'Rental'})


# In[10]:


ems = pd.read_csv(os.path.join("..", "data", "EMS Fleet Data Sep 2022.csv"), sep=';')

colnames = ['Reg No', 'Old Reg', 'Site', 'District']

# Select columns
ems = ems[colnames]

# Rename the rental column from " Rental " to "Rental"
ems = ems.rename(columns={' Rental ': 'Rental'})


# ## 2.2 Combine the white and ems fleet register datasets

# In[11]:


white_ems = pd.concat([white, ems], ignore_index=True)


# In[12]:


cols = ['Site', 'District']

# Reshaping the DataFrame for 'New' IDs
new_regs = white_ems.melt(id_vars=cols, 
                  value_vars=['Reg No'], 
                  var_name='Type', 
                  value_name='Reg').drop(columns=['Type'])

# Reshaping the DataFrame for 'Old' IDs
old_regs = white_ems.melt(id_vars=cols, 
                  value_vars=['Old Reg'], 
                  var_name='Type', 
                  value_name='Reg').drop(columns=['Type'])

# Concatenating the two DataFrames
reshaped_white_ems = pd.concat([new_regs, old_regs]).sort_values(by='Reg').reset_index(drop=True)


# In[17]:


reshaped_white_ems.shape


# In[15]:


# Create a new dataframe with only the first 5 rows
trans_df_for_report = reshaped_white_ems.head()

# Step 1: Identify unique registration numbers in the subset
unique_reg_nums_subset = sorted(trans_df_for_report['Reg'].unique())

# Step 2: Create a mapping from each unique registration number in the subset to an anonymized value
anonymized_mapping_subset = {reg_num: 'ANON_REG_NUM_' + str(i) for i, reg_num in enumerate(unique_reg_nums_subset, start=1)}

# Step 3: Apply the mapping to your subset
trans_df_for_report['Reg'] = trans_df_for_report['Reg'].map(anonymized_mapping_subset)

# Save the anonymized dataset to a new CSV file
trans_df_for_report.to_csv('../data/anonymized_raw_ems_white_register.csv', index=False)


# In[13]:


# Create a new dataframe with only the first 5 rows
trans_df_for_report = full_register.head()

# Step 1: Identify unique registration numbers in the subset
unique_reg_nums_subset = sorted(trans_df_for_report['Reg No'].unique())

# Step 2: Create a mapping from each unique registration number in the subset to an anonymized value
anonymized_mapping_subset = {reg_num: 'ANON_REG_NUM_' + str(i) for i, reg_num in enumerate(unique_reg_nums_subset, start=1)}

# Step 3: Apply the mapping to your subset
trans_df_for_report['Reg No'] = trans_df_for_report['Reg No'].map(anonymized_mapping_subset)

# Save the anonymized dataset to a new CSV file
trans_df_for_report.to_csv('../data/anonymized_raw_fleet_register.csv', index=False)


# In[14]:


trans_df_for_report


# ## 2.3 Merge the datasets

# In[29]:


raw_data = pd.merge(raw_trans, full_register, how='left', left_on='REG_NUM', right_on='Reg No')
raw_data = raw_data.drop(columns=['Reg No'])


# In[30]:


# left join df and white on REG_NUM and Reg No
raw_data = pd.merge(raw_data, reshaped_white_ems, how='left', left_on='REG_NUM', right_on='Reg')

# drop the Reg column
raw_data = raw_data.drop(columns=['Reg'])

# check for null values
print(raw_data.isnull().sum())


# ## 3. Clean the columns and remove all unnecessary transactions

# ## 3.1 Find all the vehicles in the transaction dataset that are not in the fleet register dataset

# In[31]:


# Remove all rows with null where the "MODEL DERIVATIVE" is null - vehicles under invstigation and not included in analysis
clean_data = raw_data.dropna(subset=['MODEL DERIVATIVE'])

# check for null values
print(clean_data.isnull().sum())


# In[32]:


clean_data.shape


# ## 3.2 Vehicle make

# In[33]:


# Check which "VEHICLE MAKE" and "MODEL DERIVATIVE" are null
clean_data[clean_data['VEHICLE MAKE'].isnull()]


# In[34]:


# Remove the trucks and tankers from the dataset
clean_data = clean_data[~clean_data['VEHICLE MAKE'].isnull()]

# check for null values
print(clean_data.isnull().sum())


# In[35]:


# Check the unique VEHICLE MAKE values
clean_data['VEHICLE MAKE'].unique()


# In[36]:


# Create a list of makes to check
makes_to_check = ['HINO', 'UD TRUCKS', 'IVECO', 'MITSUBISHI', 'TATA', 'NISSAN', 'ROSENBAUER', 'MITSUBISHI FUSO']

# For each of the makes to remove, display the unique models
pd.DataFrame(clean_data[clean_data['VEHICLE MAKE'].isin(makes_to_check)].groupby('VEHICLE MAKE')['MODEL DERIVATIVE'].unique())


# In[37]:


models_to_remove = ['IVECO', 'MITSUBISHI FUSO', 'ROSENBAUER', 'UD TRUCKS', 'HINO']

# Remove all of the makes to remove
clean_data = clean_data[~clean_data['VEHICLE MAKE'].isin(models_to_remove)]


# In[38]:


# Display all unique model derivatives
clean_data['MODEL DERIVATIVE'].unique()


# In[39]:


clean_data.head()


# In[40]:


# Convert all model derivatives to uppercase
clean_data['MODEL DERIVATIVE'] = clean_data['MODEL DERIVATIVE'].str.upper()


# In[41]:


# Check the unique model derivatives that contain "UD"
clean_data[clean_data['MODEL DERIVATIVE'].str.contains('UD')]['MODEL DERIVATIVE'].unique()


# In[42]:


# Remove all rows where the model derivative contains "TRUCK", "NPR", "NQR", "MASSEY" or "AXOR"
clean_data = clean_data[~clean_data['MODEL DERIVATIVE'].str.contains('TRUCK|NPR|NQR|MASSEY|AXOR|BUS|FTR')]

# Remove all rows where the model derivative is 'NISSAN UD 40A F/C C/C'
clean_data = clean_data[~clean_data['MODEL DERIVATIVE'].str.contains('NISSAN UD 40A F/C C/C')]


# In[43]:


# Remove the make and model columns
clean_data = clean_data.drop(columns=['Make', 'Model'])


# In[44]:


print(clean_data.isnull().sum())


# ## 3.2 Purchase Category

# In[45]:


# Remove all non-fuel transactions
clean_data = clean_data[clean_data['Purchase Category'] == 'FUEL']


# In[46]:


clean_data.shape


# In[47]:


# Remove the purchase category column
clean_data = clean_data.drop(columns=['Purchase Category'])


# In[48]:


print(clean_data.isnull().sum())


# ## 3.3 Merchant Names

# In[49]:


from fuzzywuzzy import process
import pandas as pd

def find_similar_names(name, names, threshold=95):
    """
    Finds similar names in a list of names.
    
    :param name: The name to find similarities to.
    :param names: The list of names to search in.
    :param threshold: The similarity threshold (0-100). Names with a similarity score above this threshold will be considered similar.
    :return: A list of similar names.
    """
    similar_names = process.extractBests(name, names, score_cutoff=threshold)
    # Filter out exact matches (similarity score of 100)
    return [sim_name for sim_name, score in similar_names if score < 100]

def consolidate_names(names_dict):
    """
    Consolidates similar names into a single representative name.

    :param names_dict: A dictionary where keys are original names and values are lists of similar names.
    :return: A dictionary mapping each name (including similar ones) to a single representative name.
    """
    consolidated_dict = {}
    for original_name, similar_names in names_dict.items():
        # Include the original name itself in the mapping
        consolidated_dict[original_name] = original_name
        for similar_name in similar_names:
            # Map similar names to the original name
            consolidated_dict[similar_name] = original_name
    return consolidated_dict

def replace_names(df, names_map, var='Merchant Name'):
    """
    Replace names in the dataframe using a mapping dictionary.

    :param df: The DataFrame containing the names.
    :param names_map: A dictionary mapping each name to a representative name.
    :return: DataFrame with replaced names.
    """
    # Replace names in the DataFrame using the mapping
    df[var] = df[var].map(names_map).fillna(df[var])
    return df


# In[113]:


# Extract unique merchant names
unique_names = clean_data['Merchant Name'].unique()

# Remove null values
unique_names = unique_names[~pd.isnull(unique_names)]

# Dictionary to hold each name and its similar names
similar_names_dict = {}

# Iterate over each unique name
for name in unique_names:
    # Find similar names
    similar_names = find_similar_names(name, unique_names)
    # Store in the dictionary
    similar_names_dict[name] = similar_names

# Print the results, excluding exact matches
for name, similarities in similar_names_dict.items():
    if similarities:  # Only print if there are non-exact matches
        print(f"{name}: {similarities}")


# In[114]:


# Consolidate similar names into a single representative name for each group
names_map = consolidate_names(similar_names_dict)

# Replace names in the DataFrame
clean_data = replace_names(clean_data, names_map)

# View the replaced DataFrame
clean_data.head()


# In[115]:


# Convert the merchant names to camel case
clean_data['Merchant Name'] = clean_data['Merchant Name'].str.title()


# ## 3.4 Transaction Amount

# In[116]:


# Check for negative transaction amounts - check these for later
clean_data[clean_data['Transaction Amount'] < 0]


# ## 3.5 Department

# In[117]:


# Check for all unique Department values
clean_data['DEPARTMENT'].unique()


# In[118]:


# Trim the white space from the end of the Department values
clean_data['DEPARTMENT'] = clean_data['DEPARTMENT'].str.strip()


# ## 3.6 Rate Card Category

# In[119]:


# Display all unique rate card categories
clean_data['RATE CARD CATEGORY'].unique()


# In[120]:


# Check how many transactions were for the 'MANAGED MAINTENANCE' rate card category
clean_data[clean_data['RATE CARD CATEGORY'] == 'MANAGED MAINTENANCE'].shape


# In[121]:


# Select a few random transactions for the managed maintenance rate card category
clean_data[clean_data['RATE CARD CATEGORY'] == 'MANAGED MAINTENANCE'].sample(10)


# Most seem to be fine and operating, although the fleet register dataset is from a later date and the vehicles could have been placed in managed maintenance by that point.

# ## 3.7 District

# In[122]:


# Display all unique District values
clean_data['District'].unique()


# In[123]:


# Check the distribution of the District variable
clean_data['District'].value_counts(dropna=False)


# In[124]:


# Change the NA district to "UNKNOWN"
clean_data['District'] = clean_data['District'].fillna('UNKNOWN')


# In[125]:


# Check the distribution of the District variable
clean_data['District'].value_counts(dropna=False)


# ## 3.8 Site

# In[126]:


# Check the distribution of the Site variable
clean_data['Site'].value_counts(dropna=False)


# In[127]:


# Change the NA site values to "UNKNOWN"
clean_data['Site'] = clean_data['Site'].fillna('UNKNOWN')


# In[128]:


# Convert all sites to upper case
clean_data['Site'] = clean_data['Site'].str.upper()


# In[129]:


# Extract unique site names
unique_names = clean_data['Site'].unique()

# Remove null values
unique_names = unique_names[~pd.isnull(unique_names)]

# Dictionary to hold each name and its similar names
similar_names_dict = {}

# Iterate over each unique name
for name in unique_names:
    # Find similar names
    similar_names = find_similar_names(name, unique_names)
    # Store in the dictionary
    similar_names_dict[name] = similar_names

# Print the results, excluding exact matches
for name, similarities in similar_names_dict.items():
    if similarities:  # Only print if there are non-exact matches
        print(f"{name}: {similarities}")


# In[130]:


# Consolidate similar names into a single representative name for each group
names_map = consolidate_names(similar_names_dict)

# Replace names in the DataFrame
clean_data = replace_names(clean_data, names_map, var='Site')

# View the replaced DataFrame
clean_data.head()


# In[131]:


# Convert the names to camel case
clean_data['Site'] = clean_data['Site'].str.title()
clean_data.head()


# In[132]:


def replace_site_names(df, site_replacements):
    """
    Replace site names in the dataframe based on certain patterns.

    :param df: DataFrame containing the 'Site' column.
    :return: DataFrame with updated 'Site' column.
    """

    # Temporarily fill NaN values with an empty string for comparison
    df['Site'] = df['Site'].fillna('')

    for old_site, new_site in site_replacements.items():
        # Check that the values are not null
        if not pd.isnull(old_site) and not pd.isnull(new_site):
            # Using str.startswith() to match the beginning of the string
            df.loc[df['Site'].str.startswith(old_site), 'Site'] = new_site

    # Revert the empty strings back to NaN
    df['Site'].replace('', pd.NA, inplace=True)

    return df


# In[133]:


site_replacements = {
    '0': 'UNKNOWN',
    '43501': 'UNKNOWN'
}

# Apply the function to replace model names
clean_data = replace_site_names(clean_data, site_replacements)

# Example of the replaced DataFrame
clean_data.head()


# In[134]:


# Check the final dimensions of the dataset
clean_data.shape


# In[135]:


# Save the cleaned dataset to a new CSV file
clean_data.to_csv(os.path.join("..", "data", "cleaned_data.csv"), index=False)


# # 4. Add the coordinates of the transactions based on the merchant name

# In[68]:


# Read in the clean dataset 
clean_data = pd.read_csv(os.path.join("..", "data", "cleaned_data.csv"))


# In[136]:


from geopy.geocoders import GoogleV3
import configparser
import pandas as pd

# Read API key from config file
config = configparser.ConfigParser()
config.read('../config.ini')
api_key = config['DEFAULT']['GOOGLE_API_KEY']

# Create a geocoder object
geolocator = GoogleV3(api_key=api_key)

# Function to get coordinates
def get_coordinates(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error occurred: {e}")
    return None, None

# Caching the geocoded results
cached_site_coordinates = {}

# Iterate over unique sites
for site in clean_data['Site'].unique():
    address = f"{site}, Eastern Cape, South Africa"
    if address not in cached_site_coordinates:
        cached_site_coordinates[address] = get_coordinates(address)

# Apply cached coordinates to the DataFrame
clean_data['Site Lat'] = clean_data['Site'].map(lambda x: cached_site_coordinates[f"{x}, Eastern Cape, South Africa"][0])
clean_data['Site Long'] = clean_data['Site'].map(lambda x: cached_site_coordinates[f"{x}, Eastern Cape, South Africa"][1])


# In[137]:


clean_data.head()


# In[138]:


# Caching the geocoded results
cached_coordinates = {}

# Iterate over unique sites
for merchant in clean_data['Merchant Name'].unique():
    address = f"{merchant}, Eastern Cape, South Africa"
    if address not in cached_coordinates:
        cached_coordinates[address] = get_coordinates(address)

# Apply cached coordinates to the DataFrame
clean_data['Merchant Lat'] = clean_data['Merchant Name'].map(lambda x: cached_coordinates[f"{x}, Eastern Cape, South Africa"][0])
clean_data['Merchant Long'] = clean_data['Merchant Name'].map(lambda x: cached_coordinates[f"{x}, Eastern Cape, South Africa"][1])


# In[148]:


# Save the transformed data to a csv file in the data folder
clean_data.to_csv(os.path.join("..", "data", "cleaned_data_with_coords.csv"), index=False)


# # 5. Add the true fuel prices to the transactions

# ## 5.1 Load the fuel price dataset

# In[149]:


# Load the cleaned data with coordinates
clean_data = pd.read_csv(os.path.join("..", "data", "cleaned_data_with_coords.csv"))


# ## 5.2 Create a fuel type column to store each vehicles fuel type

# In[140]:


# Check the unique vehicle model values
clean_data['MODEL DERIVATIVE'].unique()


# In[150]:


# Get the unique Model Derivatives
unique_models = clean_data['MODEL DERIVATIVE'].unique()

# Save them to a CSV file
unique_models_df = pd.DataFrame(unique_models, columns=['Model Derivative'])

# Create a new column for fuel type and assign all values to "Diesel"
unique_models_df['Fuel Type'] = 'Diesel'

# Sort the DataFrame by the model derivative
unique_models_df = unique_models_df.sort_values(by='Model Derivative')

unique_models_df.to_csv(os.path.join("..", "data", "unique_models.csv"), index=False)


# In[151]:


unique_models_df.shape


# In[154]:


petrol = ['CHEVROLET AVEO 1.6 L', 'FIAT UNO 1.2 5DR', 'FORD BANTAM 1.3I', 'FORD FIESTA 1.4I 5DR', 'FORD FIGO 1.4 AMBIENTE', 'FORD FOCUS 2.0 GDI TREND', 'FORD LASER 1.3 TRACER TONIC', 'FORD RANGER 2.5I P/U S/C', 'FORD RANGER 2.5I XL P/U D/C', 'GOLF 1.4 TSI COMFORTLINE DSG (92 KW) BQ13HZ', 'HYUNDAI ACCENT 1.6 FLUID 5DR', 'HYUNDAI ACCENT 1.6 FLUID AT 5D', 'HYUNDAI ACCENT 1.6 GL', 'HYUNDAI ACCENT 1.6 GLIDE', 'HYUNDAI ACCENT 1.6 GLS', 'HYUNDAI ACCENT 1.6 GLS A/T', 'HYUNDAI ACCENT 1.6 MANUAL', 'IKON 1.6 AMBIENTE',
          'ISUZU KB 160 FLEETSIDE', 'JEEP GRAND CHEROKEE 5.7 V8 O/L', 'M-BENZ GLE 500 4MATIC', 'MAHINDRA XYLO 2.5 CRDE E2 8 SE', 'NISSAN 1400 STD 5 SPEED P/U', 'NISSAN ALMERA 1.6 COMFORT', 'NISSAN HARDBODY 2.0I LWB (K08)', 'NISSAN HARDBODY 2.4I HIRIDER D', 'NISSAN HARDBODY 2.4I HR DC 4X4', 'NISSAN HARDBODY 2.4I LWB 4X4 P', 'NISSAN HARDBODY 2.4I LWB H/R K', 'NISSAN HARDBODY 2000 I LWB', 'NISSAN HARDBODY 2000I SWB', 'NISSAN HARDBODY 2400I 4X4', 'NISSAN HARDBODY 2400I LWB', 'NISSAN HARDBODY 2400I SE HIRID', 'NISSAN HARDBODY NP300 2.0I LWB', 'NISSAN HARDBODY NP300 2.4I 4X4', 'NISSAN HARDBODY NP300 2.4I HI-', 'NISSAN HARDBODY NP300 2.4I LWB', 'NISSAN NV350 2.5 PETROL P/VAN PVD3 (AMBULANCE)', 'NISSAN NV350 2.5I NARROW F/C P', 'NISSAN NV350 2.5I WIDE F/C P/V', 'NISSAN NV350 2.5PETROL', 'OPEL ASTRA 1.6 ESSENTIA', 'OPEL CORSA 1.4 COLOUR 3DR', 'OPEL CORSA CLASSIC 1.6 COMFORT', 'OPEL CORSA LITE PLUS A/C', 'OPEL CORSA UTILITY 1.8I', 'POLO VIVO 1.6 HIGHLINE HATCH BACK',
          'TATA INDICA 1.4 DLS', 'TATA INDICA 1.4 LSI', 'TOYOTA AVANZA 1.5 SX', 'TOYOTA COROLLA QUEST 1.6', 'TOYOTA COROLLA QUEST 1.6 A/T', 'TOYOTA ETIOS SEDAN 1.5 XI SD', 'TOYOTA PRADO VX 4.0 V6 A/T', 'TOYOTA QUANTUM 2.7 14 SEAT', 'TOYOTA QUANTUM 2.7 LWB P/V', 'TOYOTA QUANTUM 2.7 SESFIKILE 16S', 'TOYOTA YARIS T3 A/C H/B', 'VW CADDY KOMBI 1.6I', 'VW CITI CHICO 1400', 'VW CITI SPORT 1.4I', 'VW GOLF 2.0 TSI GTI DSG', 'VW GOLF VII 1.4 TSI COMFORTLIN', 'VW POLO 1.6 SEDAN', 'VW POLO 1.6 SEDAN AUTOMATIC', 'VW POLO VIVO 1.4', 'VW POLO VIVO 1.6 SEDAN', 'VW TENACITI 1.4I']
diesel = ['2009 TRANSPORTER', '250 D/C HI-RIDE GEN 6', '250 S/C SAFETY GEN 6', '250C S/CAB FLEETSIDE', '300 D/CAB LX 4X4', 'AMAROK 2.0 BI TDI 132 KW AUTO HIGHLINE 4 MOTION DC', 'AUDI 716-Q7 3.0D-183-QA8-7S', 'AUDI Q5 4.0L TDI QUATTRO', 'AUDI Q5 40TDI QUATTRO', 'AUDI Q7 3.0 TDI V6 QUATTRO TIP', 'B-VJ12 G02 X4 XDRIV 5 DOOR', 'BMW MOTORBIKE', 'BMW X4', 'BMW X4 XDRIVE 20D', 'BMW X4 XDRIVE 20D AT SAC', 'BMW X5 XDRIVE 30D SAV', 'CRAFTER 35 2.0 TDI 103 KW MAN MWB LCV CODE 8 (AMBULANCE)', 'CRAFTER 35 2.0 TDI 103KW MAN MWB', 'D-MAX 250 C/CAB 4X4', 'D-MAX 250 HO 4X4 CREW CAB HI-RIDER', 'D-MAX 250 HO 4X4 REGULAR CAB HI-RIDER', 'D-MAX 250 HO CREW CAB HI-RIDER', 'D-MAX 250 HO CREW CAB HI-RIDER 4X2 LCV D/CAB M180TX1', 'D-MAX 250C FLEETSIDE REGULAR CAB', 'D-MAX 250C FLEETSIDE REGULAR CAB (M180TW1)', 'D-MAX 250C REGULAR CAB FLEETSIDE REGUL', 'D-MAX 300 4X4 REGULAR CAB LX', 'DISCOVERY SPORT 132 KW 7S AUTO D', 'FORD RANGER', 'FORD RANGER 2.2 TDCI D/CAB 5MT 4X2', 'FORD RANGER 2.2 TDCI D/CAB XL 6MT 4X4 (WITH CANOPY)', 'FORD RANGER 2.2 TDCI S/CAB LR 5MT 4X2', 'FORD RANGER 2.2 TDCI S/CAB XL 6MT 4X4 (AMBULANCE)', 'FORD RANGER 2.2 TDCI XL PLUS 4', 'FORD RANGER 2.2 XL D/CAB', 'FORD RANGER 2.2TDCI 4X2 S/CAB', 'FORD RANGER 2.2TDCI L/R P/U C/', 'FORD RANGER 2.2TDCI L/R P/U D/C', 'FORD RANGER 2.2TDCI L/R P/U S/', 'FORD RANGER 2.2TDCI P/U D/C', 'FORD RANGER 2.2TDCI XL 4X4 P/U', 'FORD RANGER 2.2TDCI XL P/U D/C', 'FORD RANGER 2.2TDCI XL P/U S/C', 'FORD RANGER 2.2TDCI XL P/U SUP', 'FORD RANGER 2.2TDCI XLS P/U S/', 'FORD RANGER 2200 LWB XL', 'FORD RANGER 3.2TDCI WILDTRACK', 'FORD RANGER 3.2TDCI XLS P/U S/', 'FORD RANGER D/C', 'FORD RANGER D/CAB', 'FORD RANGER D/CAB 4X4', 'FORD RANGER D/CAB XL 4X4HR', 'HARDBODY 2.7 LWB S/CAB 4X2', 'HILUX 2.4 GD 5MT A/C S/C', 'HILUX 2.4 GD-6 4X4 SR MT A12 S/C', 'HILUX 2.4 GD-6 RB SRX MT S/C', 'HILUX 2.4 GD-6 RB SRX MT S/C', 'HILUX 2.4 GD-6 RB SRX MT S/C', 'HILUX 2.4GD-6SR 4X4 D/C', 'HILUX DC 2.4GD6 4X4 SR MT', 'HILUX DC 2.4GD6 4X4 SRX MT', 'HILUX DC 2.4GD6 RB SRX MT', 'HILUX SC 2.0 VVTI S', 'HILUX SC 2.4 GD 5MT A/C', 'HILUX SC 2.4GD6 4X4 SR MT', 'HILUX SC 2.4GD6 RB SRX MT', 'HILUXDC 2.4GD6 RB SR MT', 'HILUXSC 2.4GD S A/C 5MT', 'INTERSTER 2.5 MWB PANEL VAN',
          'ISUZU D-MAX 250 HO 4X4 REGULAR CAB HI-RIDER AMBULANCE', 'ISUZU D-MAX 250C REGULAR CAB FLEETSIDE', 'ISUZU KB 300 TDI FLEETSIDE', 'ISUZU KB 300 TDI LX', 'K40 NISSAN NP300 2.5 TDI LWB S/CAB', 'KB 250 D-TEQ HO LE PU D/', 'KB 250 HO 4X4 CREW CAB HI-RIDER', 'KB 250 HO 4X4 REGULAR CAB HI-RIDER', 'KB 250 HO CREW CAB HI-RIDER', 'KB 250 HO HI-RIDER CREW CAB', 'KB 250C BASE REGULAR CAB', 'KB 250C REGULAR CAB FLEETSIDE', 'LAND CRUSER PRADO', 'LAND ROVER DISCOVERY SPORT 132 KW 7S AUTO D', 'LANDCRUISER 79 PICK UP 4.2 DIESEL S/C', 'M-BENZ 309D/36CDI P/VAN SR 4X2', 'M-BENZ 416 CDI PV HR', 'M-BENZ GLE 350D 4MATIC', 'M-BENZ SPRINTER 515 CDI PV', 'M-BENZ VITO 113 CDI F/C P/V', 'NISSAN 2.4 DC 4X4 HR+ABS', 'NISSAN 2.5 4X4 S/C (MOBILE CLINIC)', 'NISSAN 2.5D SE+SC 4X4 ABS', 'NISSAN HARDBODY', 'NISSAN HARDBODY 2.5 TDI HIRIDE', 'NISSAN HARDBODY 2.5 TDI LWB S', 'NISSAN HARDBODY 2.5 TDI LWB SE', 'NISSAN HARDBODY 2.5TDI LWB K03', 'NISSAN HARDBODY 2700 D 4X2 D/C', 'NISSAN HARDBODY 2700D LWB', 'NISSAN HARDBODY NP300 2.5 TDI', 'NISSAN HARDBODY NP300 2.5TDI H', 'NISSAN INTERSTAR 2.5 DCI SR', 'NISSAN K36 NP300 2.4 HI RIDER D/C (RESPONSE)', 'NISSAN NP200 1.6 A/C SAFETY P', 'NISSAN NV350 2.5 16 SEAT IMPEN', 'NISSAN PATROL 3.0 TDI 4X4 P/U', 'NISSAN PATROL 3.0DI GL', 'NP300 2.0 SC 4X2', 'QUANTUM 2.7 10S COMMUTER',
          'RANGER 2.2 D/C XL 6MT 4X2', 'RANGER 2.2 TDC I DOUBLE CAB XL 6MT', 'RANGER 2.2 TDCI XL 6MT 4X2 S/C', 'RANGER 2.2 XL D/C 4X4', 'RANGER 2.2D 118KW 6MT 4X4 HR D/CAB', 'RANGER 2.2D XL 6MT 4X2 S/C', 'RANGER 2019 5MY D/C XL 2.2D 118KW 6AT 4X2HR', 'RANGER 2019 5MY D/C XL 2.2D 118KW 6AT 4X4HR', 'RANGER 2019 5MY REGULAR CAB XL 2.2D 118 KW 6MT 4X4HR', 'RANGER 2019 MY DBL CAB XL 2.2D118KW 6MT 4X4HR - RESPONSE', 'RANGER D/C 4X4HR', 'RANGER DC 2.2D 4X2 5MT 88KW', 'RANGER REGULAR CAB 4X2 HR', 'REG CAB XL 2.2D 118 KW 6MT 4X2 HR', 'TOYOTA FORTUNER 2.4 GD-6 RB AT', 'TOYOTA HILUX 2.0 VVTI P/U S/C', 'TOYOTA HILUX 2.4 GD 5MT A/C S/C', 'TOYOTA HILUX 2.4 GD-6 RB SRX', 'TOYOTA HILUX 2.4 GD-6 SR 4X4', 'TOYOTA HILUX 2.4 GD-6 SRX 4X4', 'TOYOTA HILUX 2.4GD-6SR 4X4 DC', 'TOYOTA HILUX 2.5 D-4D P/U S/C', 'TOYOTA HILUX 2.5 D-4D SRX 4X4', 'TOYOTA HILUX 2.5 D-4D SRX R/B', 'TOYOTA HILUX 2200 LWB P/U (P)', 'TOYOTA HILUX 4X4 D/CAB', 'TOYOTA HILUX S/C 2.4 GD BMT A/C', 'TOYOTA LAND CRUISER 79 PICK UP 4.2 DIESEL S/C', 'TOYOTA QUANTUM 2.5 D-4D LWB PV', 'TOYOTA QUANTUM 2.5 D-4D SESFIK', 'VN751-PANEL VAN 1.9D-077-FM5-LN30', 'VW AMAROK 2.0 TRENDLINE - ACD7 D/C', 'VW AMAROK 2.0TDI TREND', 'VW CADDY 2.0TDI (81KW) F/C C/C', 'VW CADDY MAXI 2.0TDI (81KW)', 'VW CRAFTER', 'VW CRAFTER 35 2.0 TDI 80KW F/C', 'VW CRAFTER 35 2.0 TDI MWB 16 SEATER', 'VW CRAFTER 35 2.0 TDI MWB AMBULANCE', 'VW CRAFTER 35. 20 TDI MWB AMBULANCE', 'VW CRAFTER 50 2.0 BITDI HR 120', 'VW CRAFTER 50 2.0 TDI 120 KW XLWB 23S', 'VW CRAFTER 50 2.0 TDI 120 KW XLWB PANEL VAN', 'VW CRAFTER 50 2.0 TDI HR 80KW', 'VW CRAFTER 50 2.0 TDI XLWB 23 SEATER', 'VW CRAFTER 50 2.0 TDI XLWB AMBULANCE', 'VW CRAFTER PANEL VAN', 'VW CRAFTER PANEL VAN AMBULANCE', 'VW GOLF & GTI 2.0 DSG 169 KW', 'VW JETTA VI 1.6 TDI COMFORT', 'VW VN-AMAROK 2.0D-103-HM6-282']

# Check the length of the petrol and diesel lists
len(petrol) + len(diesel)


# In[155]:


# Create a new fuel type column in clean_data
clean_data['Fuel Type'] = 'Diesel'

# Replace the fuel type for the petrol vehicles
clean_data.loc[clean_data['MODEL DERIVATIVE'].isin(petrol), 'Fuel Type'] = 'Petrol'


# In[156]:


# Save the transformed data to a csv file in the data folder
clean_data.to_csv(os.path.join("..", "data", "cleaned_data_with_coords_and_fuel_type.csv"), index=False)


# ## Add the actual fuel prices

# In[157]:


# Read in the dataset
clean_data = pd.read_csv(os.path.join("..", "data", "cleaned_data_with_coords_and_fuel_type.csv"))

# Convert the transaction date to datetime
clean_data['Transaction Date'] = pd.to_datetime(clean_data['Transaction Date'])


# In[158]:


file_path_fuel = os.path.join("..", "data", "FuelPricesWithDates.csv")

# Load the dataset
fuel_data = pd.read_csv(file_path_fuel, delimiter=';').transpose()


# Resetting the index to make the first row as header
fuel_data.reset_index(inplace=True)
new_header = fuel_data.iloc[0] # grab the first row for the header
fuel_data = fuel_data[1:] # take the data less the header row
fuel_data.columns = new_header # set the header row as the dataframe header


# Rename the columns
fuel_data.rename(columns={'95 LRP (c/l)': 'Petrol',
                          'Diesel 0.05% (c/l) ': 'Diesel',
                          '95 ULP (c/l) *': 'Petrol Inland',
                          'Diesel 0.05% (c/l) **': 'Diesel Inland'}, inplace=True)

# Change the Date column to datetime
fuel_data['Date'] = pd.to_datetime(fuel_data['Date'])

# Displaying the first few rows of the transposed dataset
fuel_data.head()


# In[159]:


# Swap out all ',' for '.' in the 'Petrol' and 'Diesel' column
fuel_data['Petrol'] = fuel_data['Petrol'].str.replace(',', '.').str.strip()
fuel_data['Diesel'] = fuel_data['Diesel'].str.replace(',', '.').str.strip()
fuel_data['Petrol Inland'] = fuel_data['Petrol Inland'].str.replace(',', '.').str.strip()
fuel_data['Diesel Inland'] = fuel_data['Diesel Inland'].str.replace(',', '.').str.strip()

# Convert the columns to the correct format
fuel_data['Date'] = pd.to_datetime(fuel_data['Date'])
fuel_data['Petrol'] = pd.to_numeric(fuel_data['Petrol']) / 100
fuel_data['Diesel'] = pd.to_numeric(fuel_data['Diesel']) / 100
fuel_data['Petrol Inland'] = pd.to_numeric(fuel_data['Petrol Inland']) / 100
fuel_data['Diesel Inland'] = pd.to_numeric(fuel_data['Diesel Inland']) / 100


# In[160]:


fuel_data


# In[166]:


def get_fuel_price(transaction_date, fuel_type, fuel_prices_df, inland=False):
    # Filter the prices up to the transaction date
    relevant_prices = fuel_prices_df[fuel_prices_df['Date'] <= transaction_date].iloc[-1]
    
    if inland:
        if fuel_type == 'Petrol':
            return relevant_prices['Petrol Inland']
        else:
            return relevant_prices['Diesel Inland']

    # Select the appropriate fuel price based on vehicle category
    if fuel_type == 'Petrol':
        return relevant_prices['Petrol']
    else:
        return relevant_prices['Diesel']


# In[167]:


clean_data['Coastal Petrol'] = clean_data.apply(
    lambda row: get_fuel_price(row['Transaction Date'], 'Petrol', fuel_data), 
    axis=1
)

clean_data['Inland Petrol'] = clean_data.apply(
    lambda row: get_fuel_price(row['Transaction Date'], 'Petrol', fuel_data, inland=True), 
    axis=1
)

clean_data['Coastal Diesel'] = clean_data.apply(
    lambda row: get_fuel_price(row['Transaction Date'], 'Diesel', fuel_data), 
    axis=1
)

clean_data['Inland Diesel'] = clean_data.apply(
    lambda row: get_fuel_price(row['Transaction Date'], 'Diesel', fuel_data, inland=True), 
    axis=1
)

clean_data.head()


# In[168]:


# Create the estimated price per litre variable
clean_data['Estimated Price Per Litre'] = clean_data['Transaction Amount'] / clean_data['No. of Litres']


# In[173]:


# Display the Estimated ppl and the Actual ppl and inland ppl
clean_data[['Transaction Date', 'Fuel Type', 'MODEL DERIVATIVE',
            'Estimated Price Per Litre', 'Coastal Petrol', 'Inland Petrol', 'Coastal Diesel', 'Inland Diesel']].sample(15)


# In[172]:


# Save the data frame
clean_data.to_csv(os.path.join("..", "data", "Final with Coords, Fuel Type and Prices.csv"), index=False)


# ## Create a fuel efficiency dataset based on tracker data (too many issues)

# In[1]:


import pandas as pd
import os

# Read in the dataset
df = pd.read_csv(os.path.join("..", "data", "Final with Coords, Fuel Type and Prices.csv"))


# In[2]:


# Read in the tracker data
file_path = os.path.join("..", "data", "OdoValuesTracker.csv")

# Load the dataset
tracker_data = pd.read_csv(file_path)


# In[3]:


tracker_data.head()


# In[5]:


# Display records for Reg num "GGF356EC"
tracker_data[tracker_data['Reg'] == 'GGF356EC']


# In[19]:


# Check the number of unique registration numbers
tracker_data['Reg'].nunique()


# In[4]:


# Drop the Odo column and rename the CastedOdo to be Odo
tracker_data = tracker_data.drop(columns=['Odo']).rename(columns={'CastedOdo': 'Odo'})


# In[5]:


df.columns


# In[8]:


df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

# Group by 'REG_NUM' to get sum of 'No. of Litres' and date range
litres_data = df.groupby('REG_NUM').agg({'Transaction Date': ['min', 'max'],
                                        'VEHICLE MAKE': 'first',
                                        'MODEL DERIVATIVE': 'first',
                                        'DEPARTMENT': 'first',
                                        'District': 'first',
                                        'Site': 'first',
                                        'Site Lat': 'first',
                                        'Site Long': 'first',
                                        'Fuel Type': 'first',
                                        'RATE CARD CATEGORY': 'first',
                                        'Transaction Amount': ['sum', 'mean'],
                                        'No. of Litres': ['sum', 'mean']}).reset_index()

# Flatten the MultiIndex columns
litres_data.columns = ['REG_NUM', 'Min Date', 'Max Date', 'VEHICLE MAKE', 'MODEL DERIVATIVE', 'DEPARTMENT', 'District', 
                       'Site', 'Site Lat', 'Site Long', 'Fuel Type', 'RATE CARD CATEGORY',
                       'Total Transaction Amount', 'Mean Transaction Amount', 'Total No. of Litres', 'Mean No. of Litres']


# In[9]:


litres_data.head()


# In[10]:


import pandas as pd

# Convert 'Date' to datetime in both datasets
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
tracker_data['Date'] = pd.to_datetime(tracker_data['Date'])

# Ensure consistent column names for vehicle registration
df.rename(columns={'REG_NUM': 'Reg'}, inplace=True)

# Calculate the min and max dates for each vehicle in both datasets
odo_range = tracker_data.groupby('Reg')['Date'].agg(['min', 'max']).reset_index()
litres_range = df.groupby('Reg')['Transaction Date'].agg(['min', 'max']).reset_index()

# Merge to find overlapping dates
overlap_data = pd.merge(odo_range, litres_range, on='Reg', how='inner', suffixes=('_odo', '_litres'))

# Determine the largest overlapping range
overlap_data['Overlap_Start'] = overlap_data[['min_odo', 'min_litres']].max(axis=1)
overlap_data['Overlap_End'] = overlap_data[['max_odo', 'max_litres']].min(axis=1)

# Filter out non-overlapping entries
overlap_data = overlap_data[overlap_data['Overlap_Start'] <= overlap_data['Overlap_End']]

# Initialize columns for total calculations
overlap_data['Total_Km'] = 0
overlap_data['Total_Litres'] = 0

# Calculate total Km and Litres for each vehicle
for index, row in overlap_data.iterrows():
    reg = row['Reg']
    start, end = row['Overlap_Start'], row['Overlap_End']

    # Filter for each vehicle within the overlapping dates
    odo_filtered = tracker_data[(tracker_data['Reg'] == reg) & (tracker_data['Date'] >= start) & (tracker_data['Date'] <= end)]
    litres_filtered = df[(df['Reg'] == reg) & (df['Transaction Date'] >= start) & (df['Transaction Date'] <= end)]

    # Calculate total kilometers and liters
    if not odo_filtered.empty and not litres_filtered.empty:
        total_km = odo_filtered['Odo'].iloc[-1] - odo_filtered['Odo'].iloc[0]
        total_litres = litres_filtered['No. of Litres'].sum()

        # Update the overlap_data DataFrame
        overlap_data.at[index, 'Total_Km'] = total_km
        overlap_data.at[index, 'Total_Litres'] = total_litres

# Calculate KMPL
overlap_data['KMPL'] = overlap_data['Total_Km'] / overlap_data['Total_Litres']


# In[11]:


# Final dataset with overlapping date ranges and KMPL
final_data = overlap_data[['Reg', 'Overlap_Start', 'Overlap_End', 'Total_Km', 'Total_Litres', 'KMPL']].dropna()

final_data.head()


# In[12]:


final_data.shape


# In[13]:


litres_data.columns


# In[14]:


# Join the litres data to the final data
final_data = final_data.merge(litres_data[['REG_NUM', 'VEHICLE MAKE', 'MODEL DERIVATIVE',
       'DEPARTMENT', 'District', 'Site', 'Site Lat', 'Site Long', 'Fuel Type',
       'RATE CARD CATEGORY', 'Total Transaction Amount',
       'Mean Transaction Amount', 'Total No. of Litres', 'Mean No. of Litres']], 
                       left_on='Reg', right_on='REG_NUM', how='left')


# In[15]:


final_data.head()


# In[16]:


# Save the final data
final_data.to_csv(os.path.join("..", "data", "Final KMPL dataset.csv"), index=False)

