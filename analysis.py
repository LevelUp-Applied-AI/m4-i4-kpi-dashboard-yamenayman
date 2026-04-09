"""Integration 4 — KPI Dashboard: Amman Digital Market Analytics

Extract data from PostgreSQL, compute KPIs, run statistical tests,
and create visualizations for the executive summary.

Usage:
    python analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine


def connect_db():
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/amman_market")
    engine = create_engine(database_url)
    return engine

def extract_data(engine):

    df_customers = pd.read_sql("SELECT * FROM customers", engine)
    df_products = pd.read_sql("SELECT * FROM products", engine)
    df_orders = pd.read_sql("SELECT * FROM orders", engine)
    df_items = pd.read_sql("SELECT * FROM order_items", engine)
    df_orders = df_orders[df_orders['status'] != 'cancelled']
    df_items = df_items[df_items['quantity'] <= 100]
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])

    return {
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders,
        "order_items": df_items
    }

def compute_kpis(data_dict):
    orders = data_dict['orders']
    items = data_dict['order_items']
    products = data_dict['products']
    customers = data_dict['customers']

    df = items.merge(orders, on='order_id') \
              .merge(products, on='product_id') \
              .merge(customers, on='customer_id')
    
    df['total_price'] = df['quantity'] * df['unit_price']

    results = {}

    results['monthly_revenue'] = df.set_index('order_date').resample('ME')['total_price'].sum()
    results['weekly_orders'] = df.groupby(pd.Grouper(key='order_date', freq='W'))['order_id'].nunique()
    results['revenue_by_city'] = df.groupby('city')['total_price'].sum().sort_values(ascending=False)
    order_totals = df.groupby('order_id')['total_price'].sum()
    results['aov'] = order_totals.mean()
    results['revenue_by_category'] = df.groupby('category')['total_price'].sum()
    return results, df 

def run_statistical_tests(df):
    stats_results = {}

    amman = df[df['city'] == 'Amman'].groupby('order_id')['total_price'].sum()
    irbid = df[df['city'] == 'Irbid'].groupby('order_id')['total_price'].sum()
    
    t_stat, p_val = stats.ttest_ind(amman, irbid)
    stats_results['city_comparison'] = {
        'test': 'Independent T-test',
        'stat': t_stat,
        'p_value': p_val,
        'interpretation': "Significant" if p_val < 0.05 else "Not Significant"
    }

    return stats_results

def create_visualizations(kpi_results, df):
    sns.set_theme(style="whitegrid", palette="colorblind")
    
    plt.figure(figsize=(10, 6))
    kpi_results['monthly_revenue'].plot(kind='line', marker='o', color='teal')
    plt.title('Monthly Revenue Shows Steady Growth')
    plt.xlabel('Month')
    plt.ylabel('Revenue (JOD)')
    plt.savefig('output/kpi_1_monthly_revenue.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=kpi_results['revenue_by_city'].index, y=kpi_results['revenue_by_city'].values)
    plt.title('Amman Leads in Total Market Revenue')
    plt.savefig('output/kpi_3_city_revenue.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='total_price')
    plt.title('Price Distribution Varies Significantly Across Categories')
    plt.savefig('output/kpi_statistical_distribution.png')
    plt.close()

def main():
    """Orchestrate the full analysis pipeline."""
    os.makedirs("output", exist_ok=True)


    engine = connect_db()

    data_dict = extract_data(engine)

    kpi_results, df_master = compute_kpis(data_dict)

    stat_results = run_statistical_tests(df_master)

    create_visualizations(kpi_results, df_master)

    print("\n" + "="*40)
    print("ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total Revenue (Amman): {kpi_results['revenue_by_city'].get('Amman', 0):.2f} JOD")
    print(f"Average Order Value (AOV): {kpi_results['aov']:.2f} JOD")
    
    city_test = stat_results['city_comparison']
    print(f"Statistical Test (Amman vs Irbid): {city_test['interpretation']} (p={city_test['p_value']:.4f})")
    print("="*40)
    print("Success! Check the 'output/' directory for charts.")

if __name__ == "__main__":
    main()