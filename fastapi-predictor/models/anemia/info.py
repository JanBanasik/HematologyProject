#!/usr/bin/env python3
"""
Script to analyze the CBC-anemia dataset and output key insights:
  - Unique classes and their counts
  - Overall descriptive statistics
  - Descriptive statistics by class
  - Correlation matrix heatmap

Usage:
  python analyze_cbc_data.py --input cbc_anemia_combined.csv --output report.txt
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURES = ['HCT','HGB','MCH','MCHC','MCV','PLT','RBC','RDW','WBC']
LABEL_COL = 'label'


def analyze(df, report_fp):
    with open(report_fp, 'w') as f:
        f.write('=== CBC-Anemia Data Analysis Report ===\n')
        f.write(f'Total samples: {len(df)}\n\n')

        # Unique classes
        classes = df[LABEL_COL].unique()
        f.write('Unique classes and counts:\n')
        counts = df[LABEL_COL].value_counts()
        for cls, cnt in counts.items():
            f.write(f'  - {cls}: {cnt}\n')
        f.write('\n')

        # Overall statistics
        f.write('Overall descriptive statistics:\n')
        desc = df[FEATURES].describe().T
        f.write(desc.to_string())
        f.write('\n\n')

        # Stats by class
        f.write('Descriptive statistics by class:\n')
        for cls in classes:
            f.write(f'-- {cls} --\n')
            d = df[df[LABEL_COL]==cls][FEATURES].describe().T
            f.write(d.to_string())
            f.write('\n')

    # Correlation heatmap
    corr = df[FEATURES].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    heatmap_fp = os.path.splitext(report_fp)[0] + '_corr.png'
    plt.tight_layout()
    plt.savefig(heatmap_fp)
    print(f'Report saved to {report_fp}')
    print(f'Correlation heatmap saved to {heatmap_fp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to cbc_anemia_combined.csv')
    parser.add_argument('--output', required=True, help='Path to write analysis report (txt)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    analyze(df, args.output)