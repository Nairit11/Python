#Import required libraries
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

#Read data
df = pd.read_csv('F:\Pokemon.csv', index_col=0)

#To plot a scatter plot
sns.lmplot(x='Attack', y='Defense', data=df)
#Can also be written as sns.lmplot(x=df.attack,y=df.Defense)

#Customize using seaborn and matplotlib
sns.lmplot(x='Attack', y='Defense', data=df, fit_reg=False, hue='Stage')
plt.xlim(0,None)
plt.ylim(0,None)

#Box plot only the stats
stats_df=df.drop(['Total', 'Stage', 'Legendary'], axis=1)
sns.boxplot(data=stats_df)

#Violin plot with user defined colours
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
sns.violinplot(x='Type 1', y='Attack', data=df, 
               palette=pkmn_type_colors)

#Swarm plot to display indvidual observations
sns.swarmplot(x='Type 1', y='Attack', data=df, palette=pkmn_type_colors)

#To melt more than one features into one
melted_df = pd.melt(stats_df, 
                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep
                    var_name="Stat") # Name of melted variable
plt.figure(figsize=(10,6)) #Plot enlarged for better view
 
sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type 1', #Colour the observations on the basis of type
              split=True,  #Separate points by hue
              palette=pkmn_type_colors) 
plt.ylim(0, 260)
plt.legend(bbox_to_anchor=(1, 1), loc=2) #Legend is placed to RHS of plot

#Heatmap
corr = stats_df.corr() #Calculate correlations
sns.heatmap(corr)

#Histogram
sns.distplot(df.Attack)

#Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)

# Factor Plot
g = sns.factorplot(x='Type 1', 
                   y='Attack', 
                   data=df, 
                   hue='Stage',  # Color by stage
                   col='Stage',  # Separate the individual plots by stage
                   kind='swarm') # Each individual plot is a Swarmplot
g.set_xticklabels(rotation=-45)  #plt.xticks(rotation=-45) only affects the last individual plot, hence shoul be done this way

# Density Plot
sns.kdeplot(df.Attack, df.Defense)

#Joint Distribution Plot
sns.jointplot(x='Attack', y='Defense', data=df) #It is a scatter plot y-x and histograms of y and x features are placed beside respective axes
