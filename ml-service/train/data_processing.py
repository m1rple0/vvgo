import pandas as pd
import numpy as np
from datetime import timedelta

def load_data(filepath):
    df = pd.read_excel(filepath)
    
    df = df.rename(columns={
        'Customer ID': 'CustomerID',
        'Invoice': 'InvoiceNo',
        'Price': 'UnitPrice'
    })
    
    df = df[df['CustomerID'].notna()]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    return df

def compute_rfm(df):
    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'Monetary'
    })
    
    return rfm

def compute_clv(rfm):
    rfm['Frequency_adj'] = rfm['Frequency'].clip(lower=1)
    recency_factor = np.exp(-rfm['Recency'] / 365)
    rfm['CLV'] = (rfm['Monetary'] * np.log1p(rfm['Frequency_adj']) / 10) * recency_factor
    rfm['CLV_log'] = np.log1p(rfm['CLV'])
    rfm = rfm.replace([np.inf, -np.inf], 0)
    return rfm

def get_segment_thresholds(rfm):
    clv_values = rfm['CLV'].values
    return {
        'Champions': float(np.percentile(clv_values, 80)),
        'Loyal Customers': float(np.percentile(clv_values, 60)),
        'Potential': float(np.percentile(clv_values, 40)),
        'Need Attention': float(np.percentile(clv_values, 20))
    }

if __name__ == "__main__":
    df = load_data('../data/Online Retail.xlsx')
    rfm = compute_rfm(df)
    rfm = compute_clv(rfm)
    thresholds = get_segment_thresholds(rfm)
    rfm.to_csv('../data/rfm_data.csv')