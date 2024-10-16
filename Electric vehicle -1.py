#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

#from Ipython.display import Image, display
get_ipython().system('pip install bar-chart-race')


import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'C:\Users\subha\Downloads\dataset.csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.columns = df.columns.str.replace(" ","_")


# In[6]:


df.columns = df.columns.str.strip().str.lower()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.isna().sum()


# In[10]:


df.model.value_counts().reset_index().head()


# In[11]:


df.legislative_district.value_counts().reset_index().head()


# In[12]:


df.vehicle_location.value_counts().reset_index().head()


# In[13]:


df.electric_utility.value_counts().reset_index().head()


# In[14]:


import pandas as pd

# Filling missing values in 'Legislative District' with the most frequent value (mode)
df['legislative_district'] = df['legislative_district'].fillna(df['legislative_district'].mode()[0])

# Filling missing values in 'Model' with mode (most frequent model)
df['model'] = df['model'].fillna(df['model'].mode()[0])

# Filling missing values in 'Electric Utility' using backward fill
df['electric_utility'] = df['electric_utility'].fillna(method='bfill')

# If you want to fill with a custom value, for example, for Electric Utility:
# df['Electric Utility'] = df['Electric Utility'].fillna('Unknown Utility')


# In[15]:


import pandas as pd

# Assuming your dataframe is named df
mode_value = df['vehicle_location'].mode()[0]  # Get the mode of the column

# Replace NaN values with the mode
df['vehicle_location'].fillna(mode_value, inplace=True)


# In[16]:


df.isnull().sum()


# In[17]:


plt.figure(figsize=(8, 6))
sns.countplot(x='electric_vehicle_type', data=df)
plt.title('Count of Different Electric Vehicle Types')
plt.xticks
plt.show()


# In[18]:


plt.figure(figsize=(8, 6))
sns.histplot(df['electric_range'], kde=True, bins=20)
plt.title('Distribution of Electric Range')
plt.xlabel('Electric Range (Miles)')
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(df['model_year'], kde=True, bins=10)
plt.title('Distribution of Model Year')
plt.show()


# In[20]:


sns.histplot(df['legislative_district'], kde=True, bins=20)
plt.title("Distribition of Legislative Districts")
plt.xlabel("legislative_district")
plt.ylabel("Frequency")
plt.show()


# In[21]:


#bivariate
plt.figure(figsize=(8, 6))
sns.scatterplot(x='model_year', y='electric_range', data=df, hue='electric_vehicle_type')
plt.title('Electric Range vs Model Year by Vehicle Type')
plt.show()


# In[22]:


plt.figure(figsize=(20, 10))
top_makes = df['make'].value_counts().nlargest(20).index
ax = sns.boxplot(x='make', y='electric_range', data=df[df['make'].isin(top_makes)])
plt.title('Electric Range by Vehicle Make',fontsize = 22)
plt.xlabel('Make',fontsize = 22)
plt.ylabel("Electric Range",fontsize = 22)
plt.xticks(rotation=45)
plt.show()

for i, box in enumerate(ax.artists):
    y = box.get_ydata()
    median = round(data[data['make'] == top_makes[i]]['electric_range'].median(),2)
    plt.text(i, median + 10, f"{median}",ha = 'center',fontsize = 20)
plt.show()


# In[23]:


# Bivariate plot: Electric Vehicle Type vs Base MSRP
plt.figure(figsize=(8, 6))
sns.barplot(x='electric_vehicle_type', y='base_msrp', data=df)
plt.title('Average Base MSRP by Electric Vehicle Type')
plt.show()


# In[24]:


# Bivariate plot: Model Year vs Base MSRP
plt.figure(figsize=(8, 6))
sns.scatterplot(x='model_year', y='base_msrp', data=df)
plt.title('Base MSRP by Model Year')
plt.show()


# In[25]:


pip install plotly


# In[26]:


import pandas as pd
import plotly.express as px


# In[27]:


ev_count_by_state = df.groupby('state').size().reset_index(name = 'ev_count')

#creating the choropleth map
fig = px.choropleth(
      ev_count_by_state,
      locations = 'state', # columns representing state code(i.e.'FL')
      locationmode = 'USA-states', #this maps to 'US-states'
      color = 'ev_count', # color by count of electric vehicles
      color_continuous_scale ='reds',# color scale
      scope = 'usa',#restricts map to USA
      labels = {'ev_count':'Number of EVs'},# color leged
      title = 'Number of Electric Vehicles by state'
)
#Update the layout fo visualization

fig.update_layout(
    geo = dict(bgcolor = 'rgba(0,0,0,0)'),
    title_x = 0.8
)

#display map
#fig.show()


# In[28]:


#task3 
import bar_chart_race as bcr
df_counts = df.groupby(['model_year', 'make']).size().unstack(fill_value=0)


# In[29]:


bcr.bar_chart_race(
    df=df_counts,
    filename='ev_make_race.gif',  # Change output file to GIF
    title='Electric Vehicle Make Count Over Time',
    period_length=3000,
    sort ='desc',
    n_bars = 10,
    steps_per_period = 45,
    figsize=(10, 6),  # Adjust figure size (width, height)
    title_size=10,  # Title size
    bar_label_size= 10 # Speed of the animation
)


# In[30]:


from IPython.display import Image

Image("ev_make_race.gif")


# In[ ]:




