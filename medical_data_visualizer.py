import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where(df['weight'] / (df['height'] / 100) ** 2 > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df[['cholesterol','gluc']] = np.where(df[['cholesterol','gluc']] > 1,  1, 0)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    #df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']])

    df_cat = pd.melt(df, id_vars=[('cardio')], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio','variable','value'])['value'].count().rename('total')

    df_cat = df_cat.reset_index()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', hue='value', data= df_cat, col='cardio', kind='bar') 

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df['age'] = (df['age'] / 1000).astype(int)
    df_heat = df

    # Calculate the correlation matrix
    corr = df_heat.corr().__round__(1)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10,10))

    # Draw the heatmap with 'sns.heatmap()'

    sns.heatmap(corr, mask = mask, annot=True)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
