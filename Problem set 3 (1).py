#!/usr/bin/env python
# coding: utf-8

# # Question 1
# Introduction:
# Special thanks to: https://github.com/justmarkham for sharing the dataset and
# materials.
# Occupations
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called users
# Step 4. Discover what is the mean age per occupation
# Step 5. Discover the Male ratio per occupation and sort it from the most to the least
# Step 6. For each occupation, calculate the minimum and maximum ages
# Step 7. For each combination of occupation and sex, calculate the mean age
# Step 8. For each occupation present the percentage of women and men

# In[1]:


# Step 1
import pandas as pd

# Step 2 and Step 3
users = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')
print(users)
print()  #for leaving next line

# Step 4
print("The mean age per occupation is : ")
print()  # For leaving next line

print(users.groupby('occupation').age.agg(['mean']))
print()  # For leaving next line

# Step 5
print("The Male ratio per occupation and sort it from the most to the least is : ")
print()  # For leaving next line
genderCountPerOccupation = users.groupby(
    ['occupation', 'gender']).gender.count()
totalGenderCount = users.groupby('gender').gender.count()
print((genderCountPerOccupation /
       totalGenderCount).sort_values(ascending=False).xs('M', level=1))
print()  # For leaving next line

# Step 6
print("For each occupation, calculate the minimum and maximum ages is : ")
print()  # For leaving next line
print(users.groupby('occupation').age.agg(['min', 'max']))
print()  # For leaving next line

# Step 7
print("For each combination of occupation and sex, calculate the mean age is : ")
print()  # For leaving next line
print(users.groupby(['occupation', 'gender']).age.agg(['mean']))
print()  # For leaving next line

# Step 8
print("For each occupation present the percentage of women and men is : ")
print()  # For leaving next line
totalPeoplePerOccupation = users.groupby('occupation').gender.count()
print(genderCountPerOccupation*100/totalPeoplePerOccupation)
print()  # For leaving next line


# # Question 2
# Euro Teams
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called euro12
# Step 4. Select only the Goal column
# Step 5. How many team participated in the Euro2012?
# Step 6. What is the number of columns in the dataset?
# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them
# to a dataframe called discipline
# Step 8. Sort the teams by Red Cards, then to Yellow Cards
# Step 9. Calculate the mean Yellow Cards given per Team
# Step 10. Filter teams that scored more than 6 goalsStep 11. Select the teams that start
# with G
# Step 12. Select the first 7 columns
# Step 13. Select all columns except the last 3
# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[97]:


# Step 1
import pandas as pd

# Step 2 and Step 3
euro2012=pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv',sep=',')


# In[98]:


euro2012


# In[101]:


euro2012['Goals']


# In[102]:


# Step 5. How many team participated in the Euro2012?
num_teams = len(euro2012.groupby('Team').groups)
print(num_teams)


# In[103]:


# Step 6. What is the number of columns in the dataset?
cols = euro2012.shape[1]
print(cols)


# In[104]:


# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline
discipline = euro2012[['Team', 'Yellow Cards', 'Red Cards']]
print(discipline)


# In[105]:


# Step 8. Sort the teams by Red Cards, then to Yellow Cards
discipline = discipline.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=False)
print(discipline)


# In[114]:


# Step 9. Calculate the mean Yellow Cards given per Team
mean_yellow_cards_per_team = discipline.groupby('Team').agg({'Yellow Cards': 'mean'})
print(mean_yellow_cards_per_team)
mean_yellow_cards_overall = mean_yellow_cards_per_team.mean()
print(mean_yellow_cards_overall)


# In[109]:


#Step 10. Filter teams that scored more than 6 goals
filtered = euro2012[euro2012["Goals"]>6]["Team"]
print("Teams having number of goals greater than 6 are"," and ".join(list(filtered)))


# In[110]:


#Step 11. Select the teams that start with G 
G_teams=euro2012[euro2012["Team"].str.startswith('G')]["Team"]
print("Teams having starting letter G are"," and ".join(list(G_teams)))


# In[111]:


#Step 12. Select the first 7 columns 
print(euro2012.iloc[:,:7])


# In[112]:


#Step 13. Select all columns except the last 3
print(euro2012.iloc[:,:-3])


# In[113]:


#Step 14. Present only the Shooting Accuracy from England, Italy and Russia
euro2012 =euro2012.set_index("Team")
print(euro2012[(euro2012.index == "England") | (euro2012.index == "Italy") | (euro2012.index == "Russia" )]["Shooting Accuracy"])


# # Question 3
# Housing
# Step 1. Import the necessary libraries
# Step 2. Create 3 different Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000
# Step 3. Create a DataFrame by joinning the Series by column
# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it
# to 'bigcolumn'
# Step 6. Ops it seems it is going only until index 99. Is it true?
# Step 7. Reindex the DataFrame so it goes from 0 to 299
# 

# In[49]:


#Import the necessary libraries (that is pandas and numpy)
import pandas as pd
import numpy as np


#Create 3 different Series, each of length 100:
s1 = pd.Series(np.random.randint(1,5, size=(100))) #random number from 1 to 4 (5 not included)
s2 = pd.Series(np.random.randint(1,4, size=(100))) #random number from 1 to 3 (4 not included)
s3 = pd.Series(np.random.randint(10000,30001, size=(100))) #random number from 10,000 to 30,000 (30001 not included)


#Create DataFrame (df) by joinning the Series by column
df = pd.DataFrame({'s1':s1,'s2':s2,'s3':s3})

#Change the name of the columns to bedrs, bathrs, price_sqr_meter
df.rename(columns = {'s1':'bedrs', 's2':'bathrs', 's3':'price_sqr_meter'}, inplace = True)


#Create one column DataFrame (df2) with the values of the 3 Series and assign it to 'bigcolumn'

#yes it is going only until 99
df2 = pd.DataFrame({'bigcolumn': pd.concat([s1, s2, s3])})

#Reindex the DataFrame so it goes from 0 to 299 (300 not included)
df2.index = pd.RangeIndex(start=0, stop=300)


# In[50]:


df


# In[51]:


df2


# # Question 4
# Wind Statistics
# The data have been modified to contain some missing values, identified by NaN.
# Using pandas should make this exercise easier, in particular for the bonus question.
# You should be able to perform all of these operations without using a for loop or
# other looping construct.
# The data in 'wind.data' has the following format:
# Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL
# MAL
# 61 1 1 15.04 14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.04
# 61 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 17.54 13.83
# 61 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
# The first three columns are year, month, and day. The remaining 12 columns are
# average windspeeds in knots at 12 locations in Ireland on that day.
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper
# datetime index.
# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it
# and apply it.
# Step 5. Set the right dates as the index. Pay attention at the data type, it should be
# datetime64[ns].
# Step 6. Compute how many values are missing for each location over the entire
# record.They should be ignored in all calculations below.
# Step 7. Compute how many non-missing values there are in total.
# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and
# all the times.
# A single number for the entire dataset.
# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean
# windspeeds and standard deviations of the windspeeds at each location over all the
# days
# A different set of numbers for each location.
# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean
# windspeed and standard deviations of the windspeeds across all the locations at each
# day.
# A different set of numbers for each day.
# Step 11. Find the average windspeed in January for each location.
# Treat January 1961 and January 1962 both as January.
# Step 12. Downsample the record to a yearly frequency for each location.
# Step 13. Downsample the record to a monthly frequency for each location.
# Step 14. Downsample the record to a weekly frequency for each location.
# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the
# windspeeds across all locations for each week (assume that the first week starts on
# January 2 1961) for the first 52 weeks.

# In[53]:


# Step 1. Import the necessary libraries
import numpy as np
import pandas as pd
import datetime

# Step 2. Import the dataset from this address dataset="https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data"
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.

data=pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data',sep="\s+",parse_dates=[[0,1,2]])
data.head()


# In[54]:


# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.
def fix_century(x):
    year=x.year-100 if x.year > 1979 else x.year
    return datetime.date(year,x.month,x.day)
data['Yr_Mo_Dy']=data['Yr_Mo_Dy'].apply(fix_century)
data.head()


# In[55]:


#Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].
data['Yr_Mo_Dy']=pd.to_datetime(data['Yr_Mo_Dy'])
data=data.set_index('Yr_Mo_Dy')
data.head()


# In[56]:


#Step 6. Compute how many values are missing for each location over the entire record.
#They should be ignored in all calculations below.
data.isnull().sum()


# In[57]:


#Step 7. Compute how many non-missing values there are in total.
data.shape[0]-data.isnull().sum()


# In[58]:


#Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
data.mean().mean() 


# In[59]:


#Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and 
#standard deviations of the windspeeds at each location over all the days
loc_stats=pd.DataFrame()
loc_stats['min']=data.min()
loc_stats['max']=data.max()
loc_stats['mean']=data.mean()
loc_stats['std']=data.std()
loc_stats


# In[60]:


#Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and 
#standard deviations of the windspeeds across all the locations at each day.
day_stats=pd.DataFrame()
day_stats['min']=data.min(axis=1)
day_stats['max']=data.max(axis=1)
day_stats['mean']=data.mean(axis=1)
day_stats['std']=data.std(axis=1)

day_stats.head() 


# In[61]:


#Step 11. Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January.

data['date']=data.index
data['month']=data['date'].apply(lambda date:date.month)
data['year']=data['date'].apply(lambda date:date.year)
data['day']=data['date'].apply(lambda date:date.day)
january_winds=data.query('month==1')
january_winds
january_winds.loc[:,'RPT':'MAL'].mean()


# In[62]:


#Step 12. Downsample the record to a yearly frequency for each location.
data.query('month == 1 and day == 1')


# In[63]:


#Step 13. Downsample the record to a monthly frequency for each location.
data.query('day == 1')


# In[64]:


#Step 14. Downsample the record to a weekly frequency for each location.
weekly_resampled_data = data.resample('W').mean()
weekly_resampled_data


# In[65]:


#Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.
df_1961 = data[data.index < pd.to_datetime('1962-01-01')]
df_1961.resample('W').mean()
df_1961.resample('W').min()
df_1961.resample('W').max()
df_1961.resample('W').std()


# # Question 5
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called chipo.
# Step 4. See the first 10 entries
# Step 5. What is the number of observations in the dataset?
# Step 6. What is the number of columns in the dataset?
# Step 7. Print the name of all the columns.
# Step 8. How is the dataset indexed?
# Step 9. Which was the most-ordered item?
# Step 10. For the most-ordered item, how many items were ordered?
# Step 11. What was the most ordered item in the choice_description column?
# Step 12. How many items were orderd in total?
# Step 13.
# • Turn the item price into a float
# • Check the item price type
# • Create a lambda function and change the type of item price
# • Check the item price type
# Step 14. How much was the revenue for the period in the dataset?
# Step 15. How many orders were made in the period?
# Step 16. What is the average revenue amount per order?
# Step 17. How many different items are sold?

# In[3]:


# Step 1. Import the necessary libraries
import pandas as pd

# Step 2. use this data set : https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv
# Step 3. Assign it to a variable called chipo.
chipo=pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv',sep='\t')
     



# In[4]:


# get head of dataset
print(chipo.head(10))


# In[5]:


# Step 5. What is the number of observations in the dataset?
"""4622"""


# In[6]:


# get number of rows
print(chipo.shape[0])


# In[7]:


# Step 6. What is the number of columns in the dataset?
"""5"""
# get number of cols
print(chipo.shape[1])


# In[8]:


# Step 7. Print the name of all the columns.
print(chipo.columns)


# In[9]:


# Step 8. How is the dataset indexed?
""" dataset is indexed numerically from starting from 0 to 4622 in steps of 1"""
print(chipo.index)


# In[10]:


# Step 9. Which was the most-ordered item?
"""Chips and Fresh Tomato Salsa"""
print(chipo[chipo.quantity==chipo.quantity.max()].item_name)


# In[11]:


# Step 10. For the most-ordered item, how many items were ordered?
"""15 orders"""
print(chipo[chipo.quantity==chipo.quantity.max()].quantity)


# In[12]:


# Step 11. What was the most ordered item in the choice_description column?
"""[Diet Coke]"""
print(chipo.groupby('choice_description').agg({'quantity':'sum'}).sort_values(by='quantity', ascending=False).head(1).index[0])


# In[13]:


# Step 12. How many items were orderd in total?
"""4972"""
print(chipo.quantity.sum())


# In[14]:


# Step 13.
# • Turn the item price into a float
"""chipo['item_price'] = chipo['item_price'].str.replace('$', '').astype(float)"""
# • Check the item price type
"""print(chipo['item_price'].dtype)"""
# • Create lambda function and change the type of item price
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x.replace('$', '')))
# • Check the item price type
#"""float64"""
print(chipo['item_price'].dtype)


# In[86]:


# Step 14. How much was the revenue for the period in the dataset?
"""34500.16"""
print(chipo['item_price'].sum())


# In[87]:


# Step 15. How many orders were made in the period?
"""1834"""
print(len(chipo.groupby('order_id').agg({'order_id':'count'})))


# In[89]:


# Step 16. What is the average revenue amount per order?
#"""7.841911"""
print(chipo.groupby('order_id').agg({'item_price':'mean'}).mean())


# In[90]:


# Step 17. How many different items are sold?
"""50"""
print(len(chipo.groupby('item_name').agg({'item_name':'count'})))


# # Question 6
# Create a line plot showing the number of marriages and divorces per capita in the
# U.S. between 1867 and 2014. Label both lines and show the legend.
# Don't forget to label your axes!
# 

# In[117]:


import pandas as pd
import matplotlib.pyplot as plt

# read data.csv into pandas DataFrame
df = pd.read_csv('C:\\Users\\GNC\\Downloads\\us-marriages-divorces-1867-2014.csv')

df.plot(x='Year', y=['Marriages_per_1000', 'Divorces_per_1000'])

# set plot title
plt.title('Marriages and Divorces per capital in the U.S. between 1867 and 2011')

# set axis labels
plt.ylabel('Marriages and Divorces per Capita')
plt.xlabel('Year')

# show grid line (optional)
plt.grid(axis='x')

# show plot
plt.show()


# # Question 7
# Create a vertical bar chart comparing the number of marriages and divorces per
# capita in the U.S. between 1900, 1950, and 2000.
# Don't forget to label your axes!

# In[122]:


# Importing the matplotlib library
import numpy as np
import matplotlib.pyplot as plt
# Declaring the figure or the plot (y, x) or (width, height)
plt.figure(figsize=[15, 10])
# Data to be plotted
marriage = [709000, 1667000, 2315000]
divorce = [56000, 385000, 944000]

X = np.arange(len(marriage))
plt.bar(X, marriage, color = 'green', width = 0.25)
plt.bar(X + 0.25, divorce, color = 'orange', width = 0.25)
plt.legend(['Marriage', 'Divorce'])
plt.xticks([i + 0.25 for i in range(3)], ['1900', '1950', '2000'])
plt.title("Vertical Bar Chart")
plt.xlabel('Marriage-Divorce-per-Capita between 1900, 1950, and 2000')
plt.ylabel('Total')
plt.show()


# # Question 8
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort
# the actors by their kill count and label each bar with the corresponding actor's name.
# Don't forget to label your axes!

# In[116]:


import pandas as pd
import matplotlib.pyplot as plt

# read data.csv file into pandas DataFrame object called df
df = pd.read_csv('C:\\Users\\GNC\\Downloads\\actor_kill_counts.csv')

# plot data as horizontal bar graph
df.plot.barh(x='Actor', y='Count')

# set axis labels
plt.ylabel('Actor')
plt.xlabel('Kill Count')

# show vertical grid lines (optional)
plt.grid(axis='x', linestyle = '--')

# show the plot
plt.show()


# # Question 9
# Create a pie chart showing the fraction of all Roman Emperors that were
# assassinated.
# Make sure that the pie chart is an even circle, labels the categories, and shows the
# percentage breakdown of the categories.

# In[15]:


import matplotlib.pyplot as plt
import pandas as pd

roman_emperors = pd.read_csv('C:\\Users\\GNC\\Downloads\\roman-emperor-reigns.csv')
assassinated_emperors = roman_emperors[
roman_emperors['Cause_of_Death'].apply(lambda x: 'assassinated' in x.lower())]

print(assassinated_emperors)
number_assassinated = len(assassinated_emperors)
print(number_assassinated)
other_deaths = len(roman_emperors) - number_assassinated
print(other_deaths)
emperor = assassinated_emperors["Emperor"]
cause_of_death = assassinated_emperors["Cause_of_Death"]
plt.pie(range(len(cause_of_death)), labels=emperor,autopct='%1.2f%%', startangle=50, radius=0.045 * 100,rotatelabels = 270)
fig = plt.figure(figsize=[15, 20])


# # Question 10
# Create a scatter plot showing the relationship between the total revenue earned by
# arcades and the number of Computer Science PhDs awarded in the U.S. between
# 2000 and 2009.
# Don't forget to label your axes!
# Color each dot according to its year.

# In[42]:


file_url = 'C:\\Users\\GNC\\Downloads\\arcade-revenue-vs-cs-doctorates.csv'
df = pd.read_csv(file_url)
df.rename(columns = {'Total Arcade Revenue (billions)':'REVENUE','Computer Science Doctorates Awarded (US)':'AWARDS'}, inplace=True)
groups = df.groupby('Year')
for name, group in groups:
    plt.plot(group.REVENUE, group.AWARDS, marker='o', linestyle='', markersize=12, label=name)

plt.legend()
plt.xlabel("REVENUE")
plt.ylabel("AWARDS")

# show the scatter plot
plt.show()


# In[ ]:




