import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np


def show_dist(evaluation_df):
    list_of_lists = evaluation_df['cvg_avg'].values
    data = np.array(list_of_lists)

    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1  
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.7 * iqr
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]

    fig, ax = plt.subplots()

    box_plot = ax.boxplot(data, vert=False, labels=['Datos'], patch_artist=True)

    colors = ['lightblue']
    for box in box_plot['boxes']:
        box.set_facecolor(colors[0])

    label_offset = 0.1
    label_offset_q3 = 0.1
    ax.text(median, 1 + label_offset, f'Median: {median:.2f}', va='bottom', ha='center', color='blue', fontsize=7)
    ax.text(q1, 1 + label_offset_q3, f'Q1: {q1:.2f}', va='bottom', ha='right', color='green',fontsize=7)
    ax.text(q3, 1 + label_offset, f'Q3: {q3:.2f}', va='bottom', ha='left', color='red',fontsize=7)

    if len(outliers) > 0:
        ax.scatter(outliers, np.ones_like(outliers), color='orange', marker='o', label='Outliers')

    plt.xlabel('$C_j$')
    plt.yticks([])
    plt.legend()
    plt.grid(axis='x', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.show()

def compare_models_average_size(coverage_df, trials:bool=False):
    pastel_palette = sns.color_palette(['#FFCDD2', '#B39DDB', '#90CAF9', '#81C784', '#E3A3E6', '#E0B0FF'])
    fig, ax = plt.subplots(figsize=(10, 6))
    

    if trials:
        coverage_df = coverage_df.groupby(['predictor','alpha','model'])['size'].mean().reset_index()
        coverage_df.pivot_table(index=["predictor", "alpha"], columns="model", values="size").plot(kind="bar", ax=ax, color=pastel_palette)
        mean_size = coverage_df['size'].mean()
        ax.axhline(mean_size, color='red', linestyle='--', linewidth=2, label=f'Average size: {math.floor(mean_size):.2f}')
        ax.text(x=0.5, y=mean_size, s=f'{math.floor(mean_size):.2f}', color='red', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
        plt.xlabel('Predictor and significance level')
        plt.ylabel('Average size')
        plt.legend(title="Model", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        coverage_df.pivot_table(index=["predictor", "alpha"], columns="model", values="sz_avg").plot(kind="bar", ax=ax, color=pastel_palette)
        max_size = coverage_df['sz_avg'].mean()
        ax.axhline(max_size, color='red', linestyle='--', linewidth=2, label=f'Max size: {max_size:.2f}')
        ax.text(x=0.5, y=max_size, s=f'{math.floor(max_size):.2f}', color='red', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
        plt.xlabel('Predictor and significance level')
        plt.ylabel('Average size')
        plt.legend(title="Model", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def compare_models_average_coverage(coverage_df):
    pastel_palette = sns.color_palette(['#FFCDD2', '#B39DDB', '#90CAF9', '#81C784', '#E3A3E6', '#E0B0FF'])
    fig, ax = plt.subplots(figsize=(10, 6))
    coverage_df.pivot_table(index=["predictor", "alpha"], columns="model", values="cvg_avg").plot(kind="bar", ax=ax, color=pastel_palette)
    max_coverage = coverage_df['cvg_avg'].mean()
    ax.axhline(max_coverage, color='red', linestyle='--', linewidth=2, label=f'Max coverage: {max_coverage:.2f}')
    ax.text(x=0.5, y=max_coverage, s=f'{max_coverage:.2f}', color='red', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
    plt.xlabel('Predictor and significance level')
    plt.ylabel('Average coverage')
    plt.legend(title="Model", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




def graph_coverage_topk(evaluation):
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = evaluation.melt(id_vars=["model", "predictor", "alpha"], value_vars=["cvg_avg", "top5_avg", "top1_avg"])
    sns.barplot(data=pivot_df, x="predictor", y="value", hue="variable", ci=None, ax=ax)
    plt.xlabel('Predictor')
    plt.ylabel('Coverage')
    plt.legend(title="Variable")
    plt.xticks(rotation=45)

    for p in ax.patches:
        value = p.get_height()
        ax.annotate(f'{value:.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()



def size_distribution(df_size_correct, num_classes:int = 20):
    plt.figure(figsize=(10, 6))
    plt.hist(df_size_correct['size'], bins=range(0, num_classes+2), edgecolor='black', align='left')  
    plt.xlabel('Prediction set size')
    plt.ylabel('Frecuency')
    plt.xticks(range(0, num_classes+1))  

    plt.show()

def prediction_set_size_stratified_plot(data):
    data["prediction_set_average"] = [math.floor(round(float(value), 2)) for value in data["size"]]
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="difficulty", y="count", hue="prediction_set_average", palette="coolwarm",ci=None)

    plt.xlabel('Difficulty')
    plt.ylabel('Count')

    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()


def violation_plot(data):
    data["violation"] = [round(float(value), 3) for value in data["violation"]]
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(data=data, x="difficulty", y="count", hue="violation", palette="coolwarm", ci=False)

    plt.xlabel('Difficulty')
    plt.ylabel('Count')

    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points')
    plt.tight_layout()
    plt.show()
