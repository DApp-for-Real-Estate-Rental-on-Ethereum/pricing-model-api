"""
Market Trend Dashboard Module
=============================

Provides time-series analysis and clustering of rental market trends.
Generates price evolution, occupancy rates, and market insights per city/season.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import logging
import json

from deployment.db_connection import execute_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-trends", tags=["Market Trends"])


class TrendDataPoint(BaseModel):
    """Single data point in a time series."""
    
    period: str = Field(..., description="Time period (YYYY-MM)")
    avg_price_mad: float = Field(..., description="Average daily price in MAD")
    occupancy_rate: float = Field(..., ge=0.0, le=1.0, description="Occupancy rate (0-1)")
    n_bookings: int = Field(..., ge=0, description="Number of bookings")
    n_cancellations: int = Field(..., ge=0, description="Number of cancellations")
    avg_stay_length_days: float = Field(..., ge=0, description="Average stay length")


class CityTrendResponse(BaseModel):
    """Market trend data for a specific city."""
    
    city: str
    period_start: str
    period_end: str
    data_points: List[TrendDataPoint]
    trend_direction: str = Field(..., description="RISING, STABLE, DECLINING")
    price_change_percent: float = Field(..., description="Price change over period (%)")
    avg_occupancy: float = Field(..., ge=0.0, le=1.0)


class MarketInsight(BaseModel):
    """Market insight/forecast."""
    
    city: str
    insight_type: str = Field(..., description="PRICE_FORECAST, OCCUPANCY_FORECAST, SEASONALITY")
    message: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    data: Dict


class MarketTrendsResponse(BaseModel):
    """Complete market trends response."""
    
    trends: List[CityTrendResponse]
    insights: List[MarketInsight]
    generated_at: str


def get_city_trend_data(
    city: Optional[str] = None,
    period_months: int = 12,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract time-series data for market trends.
    
    Args:
        city: Optional city filter
        period_months: Number of months to analyze
        start_date: Optional start date (YYYY-MM-DD), defaults to period_months ago
        
    Returns:
        DataFrame with monthly aggregated data
    """
    logger.info(f"Analyzing market trends for City: {city}")
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start = datetime.now() - timedelta(days=period_months * 30)
    
    end = datetime.now()
    
    # Build query
    city_filter = ""
    params = [start.date(), end.date()]
    
    if city:
        city_filter = """
            AND (LOWER(p.city) = LOWER(%s) OR LOWER(a.city) = LOWER(%s))
        """
        params.append(city)
        params.append(city)
    
    query = f"""
        SELECT 
            DATE_TRUNC('month', b.check_in_date) as month,
            COALESCE(p.city, a.city, 'unknown') as city,
            COUNT(*) as n_bookings,
            SUM(CASE WHEN b.status LIKE '%%%%CANCELLED%%%%' THEN 1 ELSE 0 END) as n_cancellations,
            AVG(b.total_price / GREATEST(EXTRACT(EPOCH FROM (b.check_out_date::timestamp - b.check_in_date::timestamp)) / 86400, 1)) as avg_daily_price,
            AVG(EXTRACT(EPOCH FROM (b.check_out_date::timestamp - b.check_in_date::timestamp)) / 86400) as avg_stay_length,
            SUM(EXTRACT(EPOCH FROM (b.check_out_date::timestamp - b.check_in_date::timestamp)) / 86400) as total_nights_booked
        FROM bookings b
        JOIN properties p ON p.id = b.property_id
        LEFT JOIN addresses a ON p.address_id = a.id
        WHERE b.check_in_date >= %s
          AND b.check_in_date <= %s
          AND b.status IN ('COMPLETED', 'CONFIRMED', 'TENANT_CHECKED_OUT', 'CANCELLED_BY_TENANT')
          {city_filter}
        GROUP BY DATE_TRUNC('month', b.check_in_date), COALESCE(p.city, a.city, 'unknown')
        ORDER BY month DESC, city
    """
    
    # print(f"DEBUG QUERY: {query}")
    # print(f"DEBUG PARAMS: {params}")
    results = execute_query(query, tuple(params))
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Ensure numeric columns are float
    for col in ['avg_daily_price', 'avg_stay_length', 'total_nights_booked', 'n_bookings', 'n_cancellations']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Calculate occupancy rate (simplified: nights booked / available nights)
    # For simplicity, assume each property is available 30 days/month
    if len(df) > 0:
        # Get property count per city per month
        property_count_query = f"""
            SELECT 
                DATE_TRUNC('month', CURRENT_DATE) as month,
                COALESCE(p.city, a.city, 'unknown') as city,
                COUNT(DISTINCT p.id) as n_properties
            FROM properties p
            LEFT JOIN addresses a ON p.address_id = a.id
            WHERE p.status IN ('APPROVED', 'VISIBLE_ONLY_FOR_TENANTS')
              {city_filter.replace('b.check_in_date', 'CURRENT_DATE').replace('b.status', 'p.status') if city_filter else ''}
            GROUP BY COALESCE(p.city, a.city, 'unknown')
        """
        # Simplified: use average property count
        # In production, you'd join this properly
        
        # Approximate occupancy: nights_booked / (n_properties * 30)
        # For now, use a simplified calculation
        df['occupancy_rate'] = df['total_nights_booked'] / (df['total_nights_booked'].max() * 2) if df['total_nights_booked'].max() > 0 else 0.0
        df['occupancy_rate'] = df['occupancy_rate'].clip(0.0, 1.0)
    
    return df


def calculate_trend_direction(df: pd.DataFrame) -> str:
    """
    Determine if trend is RISING, STABLE, or DECLINING.
    
    Uses linear regression slope on price data.
    """
    if len(df) < 2:
        return "STABLE"
    
    # Sort by month
    df_sorted = df.sort_values('month')
    
    # Calculate price trend
    prices = df_sorted['avg_daily_price'].values
    if len(prices) < 2 or np.std(prices) == 0:
        return "STABLE"
    
    # Simple linear trend
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    
    # Normalize slope by average price
    avg_price = np.mean(prices)
    slope_percent = (slope / avg_price) * 100 if avg_price > 0 else 0
    
    if slope_percent > 2:
        return "RISING"
    elif slope_percent < -2:
        return "DECLINING"
    else:
        return "STABLE"


def cluster_city_trends(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster cities by trend patterns using K-Means.
    
    Features: price trend slope, occupancy trend, seasonality variance
    """
    if len(df) < n_clusters:
        return df
    
    # Group by city and calculate features
    city_features = []
    city_names = []
    
    for city in df['city'].unique():
        city_df = df[df['city'] == city].sort_values('month')
        
        if len(city_df) < 2:
            continue
        
        prices = city_df['avg_daily_price'].values
        occupancies = city_df['occupancy_rate'].values
        
        # Calculate trend slope
        x = np.arange(len(prices))
        price_slope = np.polyfit(x, prices, 1)[0] if len(prices) > 1 else 0
        occupancy_slope = np.polyfit(x, occupancies, 1)[0] if len(occupancies) > 1 else 0
        
        # Calculate variance (seasonality indicator)
        price_variance = np.var(prices) if len(prices) > 1 else 0
        
        city_features.append([price_slope, occupancy_slope, price_variance])
        city_names.append(city)
    
    if len(city_features) < n_clusters:
        return df
    
    # Normalize features
    features_array = np.array(city_features)
    features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_normalized)
    
    # Add cluster labels
    city_cluster_map = dict(zip(city_names, clusters))
    df['trend_cluster'] = df['city'].map(city_cluster_map).fillna(-1)
    
    return df


@router.get("/city/{city}", response_model=CityTrendResponse)
async def get_city_trends(
    city: str,
    period_months: int = Query(12, ge=1, le=24),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD")
):
    """
    Get market trend data for a specific city.
    
    Returns time-series data with price evolution, occupancy rates, and trend direction.
    """
    try:
        df = get_city_trend_data(city=city, period_months=period_months, start_date=start_date)
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for city: {city}")
        
        # Filter to requested city
        city_df = df[df['city'].str.lower() == city.lower()].copy()
        
        if len(city_df) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for city: {city}")
        
        # Sort by month
        city_df = city_df.sort_values('month')
        
        # Calculate trend direction
        trend_direction = calculate_trend_direction(city_df)
        
        # Calculate price change
        prices = city_df['avg_daily_price'].values
        if len(prices) >= 2:
            price_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] > 0 else 0.0
        else:
            price_change = 0.0
        
        # Format data points
        data_points = []
        for _, row in city_df.iterrows():
            month_str = row['month'].strftime('%Y-%m') if hasattr(row['month'], 'strftime') else str(row['month'])[:7]
            data_points.append(TrendDataPoint(
                period=month_str,
                avg_price_mad=float(row.get('avg_daily_price', 0.0)),
                occupancy_rate=float(row.get('occupancy_rate', 0.0)),
                n_bookings=int(row.get('n_bookings', 0)),
                n_cancellations=int(row.get('n_cancellations', 0)),
                avg_stay_length_days=float(row.get('avg_stay_length', 0.0))
            ))
        
        avg_occupancy = float(city_df['occupancy_rate'].mean())
        
        period_start = data_points[0].period if data_points else ""
        period_end = data_points[-1].period if data_points else ""
        
        return CityTrendResponse(
            city=city,
            period_start=period_start,
            period_end=period_end,
            data_points=data_points,
            trend_direction=trend_direction,
            price_change_percent=round(price_change, 2),
            avg_occupancy=round(avg_occupancy, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting city trends for {city}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/all-cities", response_model=MarketTrendsResponse)
async def get_all_cities_trends(
    period_months: int = Query(12, ge=1, le=24),
    include_insights: bool = Query(True)
):
    """
    Get market trends for all cities with clustering and insights.
    """
    try:
        df = get_city_trend_data(period_months=period_months)
        
        if len(df) == 0:
            return MarketTrendsResponse(
                trends=[],
                insights=[],
                generated_at=datetime.now().isoformat()
            )
        
        # Cluster cities by trend patterns
        df_clustered = cluster_city_trends(df, n_clusters=3)
        
        # Build response for each city
        trends = []
        cities = df['city'].unique()
        
        for city in cities:
            city_df = df[df['city'] == city].sort_values('month')
            
            if len(city_df) == 0:
                continue
            
            trend_direction = calculate_trend_direction(city_df)
            prices = city_df['avg_daily_price'].values
            price_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if len(prices) >= 2 and prices[0] > 0 else 0.0
            
            data_points = []
            for _, row in city_df.iterrows():
                month_str = row['month'].strftime('%Y-%m') if hasattr(row['month'], 'strftime') else str(row['month'])[:7]
                data_points.append(TrendDataPoint(
                    period=month_str,
                    avg_price_mad=float(row.get('avg_daily_price', 0.0)),
                    occupancy_rate=float(row.get('occupancy_rate', 0.0)),
                    n_bookings=int(row.get('n_bookings', 0)),
                    n_cancellations=int(row.get('n_cancellations', 0)),
                    avg_stay_length_days=float(row.get('avg_stay_length', 0.0))
                ))
            
            trends.append(CityTrendResponse(
                city=str(city),
                period_start=data_points[0].period if data_points else "",
                period_end=data_points[-1].period if data_points else "",
                data_points=data_points,
                trend_direction=trend_direction,
                price_change_percent=round(price_change, 2),
                avg_occupancy=round(float(city_df['occupancy_rate'].mean()), 3)
            ))
        
        # Generate insights
        insights = []
        if include_insights:
            for city in cities[:5]:  # Limit to top 5 cities
                city_df = df[df['city'] == city].sort_values('month')
                if len(city_df) < 3:
                    continue
                
                prices = city_df['avg_daily_price'].values
                occupancies = city_df['occupancy_rate'].values
                
                # Simple forecast: extrapolate trend
                if len(prices) >= 2:
                    price_trend = (prices[-1] - prices[0]) / len(prices)
                    forecast_price = prices[-1] + price_trend
                    
                    insights.append(MarketInsight(
                        city=str(city),
                        insight_type="PRICE_FORECAST",
                        message=f"Price trend suggests {forecast_price:.0f} MAD average next month",
                        confidence=0.7,
                        data={"forecast_price": float(forecast_price), "trend": float(price_trend)}
                    ))
        
        return MarketTrendsResponse(
            trends=trends,
            insights=insights,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting all cities trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/insights/{city}", response_model=List[MarketInsight])
async def get_city_insights(
    city: str,
    period_months: int = Query(12, ge=1, le=24)
):
    """
    Get AI-generated market insights for a specific city.
    
    Includes price forecasts, occupancy predictions, and seasonality analysis.
    """
    try:
        df = get_city_trend_data(city=city, period_months=period_months)
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for city: {city}")
        
        city_df = df[df['city'].str.lower() == city.lower()].sort_values('month')
        
        if len(city_df) < 2:
            return []
        
        insights = []
        
        # Price forecast
        prices = city_df['avg_daily_price'].values
        if len(prices) >= 2:
            price_trend = (prices[-1] - prices[0]) / len(prices)
            forecast_price = prices[-1] + price_trend
            
            insights.append(MarketInsight(
                city=city,
                insight_type="PRICE_FORECAST",
                message=f"Based on recent trends, expected average price: {forecast_price:.0f} MAD",
                confidence=0.75,
                data={"forecast_price": float(forecast_price), "current_price": float(prices[-1])}
            ))
        
        # Occupancy forecast
        occupancies = city_df['occupancy_rate'].values
        if len(occupancies) >= 2:
            occ_trend = (occupancies[-1] - occupancies[0]) / len(occupancies)
            forecast_occ = max(0.0, min(1.0, occupancies[-1] + occ_trend))
            
            insights.append(MarketInsight(
                city=city,
                insight_type="OCCUPANCY_FORECAST",
                message=f"Occupancy rate expected to be {forecast_occ:.1%} next month",
                confidence=0.7,
                data={"forecast_occupancy": float(forecast_occ), "current_occupancy": float(occupancies[-1])}
            ))
        
        # Seasonality
        if len(city_df) >= 6:
            # Check for seasonal patterns
            monthly_prices = city_df.groupby(city_df['month'].dt.month)['avg_daily_price'].mean()
            if len(monthly_prices) >= 3:
                peak_month = monthly_prices.idxmax()
                low_month = monthly_prices.idxmin()
                
                insights.append(MarketInsight(
                    city=city,
                    insight_type="SEASONALITY",
                    message=f"Peak season: Month {peak_month}, Low season: Month {low_month}",
                    confidence=0.8,
                    data={"peak_month": int(peak_month), "low_month": int(low_month)}
                ))
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting insights for {city}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

