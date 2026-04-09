"""Tests for the KPI dashboard analysis.

Write at least 3 tests:
1. test_extraction_returns_dataframes — extract_data returns a dict of DataFrames
2. test_kpi_computation_returns_expected_keys — compute_kpis returns a dict with your 5 KPI names
3. test_statistical_test_returns_pvalue — run_statistical_tests returns results with p-values
"""
import pytest
import pandas as pd
from analysis import connect_db, extract_data, compute_kpis, run_statistical_tests

@pytest.fixture(scope="module")
def db_data():
    
    engine = connect_db()
    data = extract_data(engine)
    return data

def test_extraction_returns_dataframes(db_data):
    """Connect to the database, extract data, and verify the result is a dict of DataFrames."""
    assert isinstance(db_data, dict), "The extraction result must be a dictionary."
    
    expected_tables = ["customers", "products", "orders", "order_items"]
    for table in expected_tables:
        assert table in db_data, f"Missing table in extracted data: {table}"
        assert isinstance(db_data[table], pd.DataFrame), f"Value for {table} is not a Pandas DataFrame"

def test_kpi_computation_returns_expected_keys(db_data):
    """Compute KPIs and verify the result contains all expected KPI names."""
    kpi_results, _ = compute_kpis(db_data)
    
    assert isinstance(kpi_results, dict), "KPI computation result must be a dictionary."
    
    expected_kpis = [
        'monthly_revenue', 
        'weekly_orders', 
        'revenue_by_city', 
        'aov', 
        'revenue_by_category'
    ]
    
    for kpi in expected_kpis:
        assert kpi in kpi_results, f"Expected KPI '{kpi}' is missing from the results."

def test_statistical_test_returns_pvalue(db_data):
    """Run statistical tests and verify results include p-values."""
    _, df_master = compute_kpis(db_data)
    
    stat_results = run_statistical_tests(df_master)
    
    assert 'city_comparison' in stat_results, "Test results missing 'city_comparison'."
    
    p_value = stat_results['city_comparison']['p_value']
    assert isinstance(p_value, float), "p-value must be a numeric float."
    assert 0 <= p_value <= 1, "p-value must be mathematically between 0 and 1."