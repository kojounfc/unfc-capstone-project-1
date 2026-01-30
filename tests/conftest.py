"""
Pytest configuration and shared fixtures for the test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_order_items():
    """Create sample order_items DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "order_id": [101, 101, 102, 103, 103],
            "product_id": [2001, 2002, 2001, 2003, 2002],
            "inventory_item_id": [3001, 3002, 3003, 3004, 3005],
            "status": ["Complete", "Returned", "Complete", "Shipped", "Cancelled"],
            "created_at": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "shipped_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", None, None]
            ),
            "delivered_at": pd.to_datetime(
                ["2024-01-04", "2024-01-04", "2024-01-05", None, None]
            ),
            "returned_at": pd.to_datetime([None, "2024-01-10", None, None, None]),
            "sale_price": [50.0, 75.0, 50.0, 100.0, 75.0],
        }
    )


@pytest.fixture
def sample_orders():
    """Create sample orders DataFrame for testing."""
    return pd.DataFrame(
        {
            "order_id": [101, 102, 103],
            "user_id": [1001, 1002, 1003],
            "status": ["Complete", "Complete", "Shipped"],
            "gender": ["F", "M", "F"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "shipped_at": pd.to_datetime(["2024-01-02", "2024-01-03", None]),
            "delivered_at": pd.to_datetime(["2024-01-04", "2024-01-05", None]),
            "returned_at": pd.to_datetime([None, None, None]),
            "num_of_item": [2, 1, 2],
        }
    )


@pytest.fixture
def sample_products():
    """Create sample products DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [2001, 2002, 2003],
            "name": ["Product A", "Product B", "Product C"],
            "category": ["Jeans", "Tops & Tees", "Outerwear & Coats"],
            "brand": ["Brand X", "Brand Y", "Brand Z"],
            "department": ["Women", "Men", "Women"],
            "cost": [20.0, 30.0, 50.0],
            "retail_price": [60.0, 80.0, 120.0],
            "sku": ["SKU001", "SKU002", "SKU003"],
            "distribution_center_id": [1, 1, 2],
        }
    )


@pytest.fixture
def sample_users():
    """Create sample users DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1001, 1002, 1003],
            "first_name": ["Alice", "Bob", "Carol"],
            "last_name": ["Smith", "Jones", "Davis"],
            "email": ["alice@test.com", "bob@test.com", "carol@test.com"],
            "age": [25, 35, 45],
            "gender": ["F", "M", "F"],
            "city": ["New York", "Los Angeles", "Chicago"],
            "state": ["NY", "CA", "IL"],
            "country": ["USA", "USA", "USA"],
            "postal_code": ["10001", "90001", "60601"],
            "latitude": [40.7128, 34.0522, 41.8781],
            "longitude": [-74.0060, -118.2437, -87.6298],
            "traffic_source": ["Search", "Organic", "Email"],
            "created_at": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-12-01"]),
        }
    )


@pytest.fixture
def sample_merged_df(sample_order_items, sample_orders, sample_products, sample_users):
    """Create a sample merged DataFrame similar to production data."""
    df = pd.DataFrame(
        {
            "order_item_id": [1, 2, 3, 4, 5],
            "order_id": [101, 101, 102, 103, 103],
            "user_id": [1001, 1001, 1002, 1003, 1003],
            "product_id": [2001, 2002, 2001, 2003, 2002],
            "product_dim_id": [2001, 2002, 2001, 2003, 2002],
            "user_dim_id": [1001, 1001, 1002, 1003, 1003],
            "item_status": ["Complete", "Returned", "Complete", "Shipped", "Cancelled"],
            "order_status": ["Complete", "Complete", "Complete", "Shipped", "Shipped"],
            "sale_price": [50.0, 75.0, 50.0, 100.0, 75.0],
            "retail_price": [60.0, 80.0, 60.0, 120.0, 80.0],
            "cost": [20.0, 30.0, 20.0, 50.0, 30.0],
            "category": [
                "Jeans",
                "Tops & Tees",
                "Jeans",
                "Outerwear & Coats",
                "Tops & Tees",
            ],
            "brand": ["Brand X", "Brand Y", "Brand X", "Brand Z", "Brand Y"],
            "traffic_source": ["Search", "Search", "Organic", "Email", "Email"],
            "user_gender": ["F", "F", "M", "F", "F"],
            "is_returned_item": [0, 1, 0, 0, 0],
            "is_returned_order": [0, 0, 0, 0, 0],
            "item_margin": [30.0, 45.0, 30.0, 50.0, 45.0],
            "item_margin_pct": [0.6, 0.6, 0.6, 0.5, 0.6],
            "discount_amount": [10.0, 5.0, 10.0, 20.0, 5.0],
            "discount_pct": [0.167, 0.0625, 0.167, 0.167, 0.0625],
            "item_delivered_at": pd.to_datetime(
                ["2024-01-04", "2024-01-04", "2024-01-05", None, None]
            ),
            "country": ["USA", "USA", "USA", "Canada", "Canada"],
        }
    )
    return df
