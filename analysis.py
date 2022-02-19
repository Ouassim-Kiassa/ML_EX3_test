import pandas as pd
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def to_tabular(path):
    df = load_files(f"{path}")

    df = pd.DataFrame([df.data, df.target.tolist()]).T
    df.columns = ['text', 'target']
    
    return df


def class_distribution(df, title):
    plt.figure(figsize=(20,8))
    ax = sns.barplot(x="target", y=df.index, data=df, palette="Blues_d")
    ax.set_title(f'Class Distribution for {title}')


def length_distribution(df):
    df['length'] = df['text'].str.len()
    plt.figure(figsize=(12.8,6))
    sns.distplot(df['length']).set_title('length distribution');
    print(df['length'].describe())

def length_distribution_after_quantile(df):
    quantile_95 = df['length'].quantile(0.95)
    df_95 = df[df['length'] < quantile_95]

    plt.figure(figsize=(12.8,6))
    sns.distplot(df_95['length']).set_title('length distribution after taking 95 percent onward for better histogram')


def barplot(df):
    plt.figure(figsize=(12.8,6))
    sns.boxplot(data=df, x='target', y='length', width=.5);


def barplot_no_long_articles(df):
    quantile_95 = df['length'].quantile(0.95)
    df_95 = df[df['length'] < quantile_95]
    plt.figure(figsize=(12.8,6))
    sns.boxplot(data=df_95, x='target', y='length');