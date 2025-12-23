#!/usr/bin/env python3
"""
JSON to CSV Pipeline - Convert Scraped Airbnb Data to Clean Dataset

This script processes all JSON files from the scraping campaign and converts
them into a single clean CSV file ready for model training.

Features:
- Processes all JSON files in raw_scrapes directory
- Extracts relevant features from nested JSON structure
- Handles missing values and data validation
- Adds seasonal and temporal features
- Deduplicates listings
- Outputs clean CSV with standardized schema

Usage:
    python json_to_csv_pipeline.py --input-dir data/raw_scrapes_expanded/by_city --output data/morocco_listings_full.csv
    python json_to_csv_pipeline.py --input-dir data/raw_scrapes_expanded/by_city --output data/morocco_listings_full.csv --deduplicate
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_to_csv.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CITY MAPPING
# ============================================================================

CITY_SLUG_MAP = {
    "agadir": "Agadir",
    "al_hociema": "Al Hoceima",
    "casablanca": "Casablanca",
    "chefchaouen": "Chefchaouen",
    "essaouira": "Essaouira",
    "fes": "Fes",
    "marrakech": "Marrakech",
    "meknes": "Meknes",
    "ouarzazate": "Ouarzazate",
    "oujda": "Oujda",
    "rabat": "Rabat",
    "tangier": "Tangier",
    "tetouan": "Tétouan",
}

# ============================================================================
# HELPER FUNCTIONS - PARSING
# ============================================================================

def parse_filename_metadata(filepath: Path) -> Dict[str, Any]:
    """
    Parse metadata from filename pattern:
    al_hociema_fall_2025_2025-09-05_3nights.json
    
    Returns:
        dict with city_slug, season, year, check_in, stay_length_nights
    """
    filename = filepath.stem
    parts = filename.split('_')
    
    metadata = {
        'city_slug': None,
        'season': None,
        'year': None,
        'check_in': None,
        'stay_length_nights': None,
        'file_source': str(filepath.relative_to(filepath.parents[4]))
    }
    
    try:
        # Extract city slug (may be multi-part like "al_hociema")
        # Find the season marker
        seasons = ['winter', 'spring', 'summer', 'fall', 'autumn']
        season_idx = None
        for i, part in enumerate(parts):
            if part.lower() in seasons:
                season_idx = i
                break
        
        if season_idx is not None:
            # City slug is everything before season
            metadata['city_slug'] = '_'.join(parts[:season_idx])
            metadata['season'] = parts[season_idx]
            
            # Year is after season
            if season_idx + 1 < len(parts) and parts[season_idx + 1].isdigit():
                metadata['year'] = int(parts[season_idx + 1])
            
            # Check-in date
            if season_idx + 2 < len(parts):
                try:
                    metadata['check_in'] = datetime.strptime(parts[season_idx + 2], '%Y-%m-%d').date()
                except ValueError:
                    pass
            
            # Stay length
            if season_idx + 3 < len(parts):
                nights_match = re.search(r'(\d+)nights?', parts[season_idx + 3])
                if nights_match:
                    metadata['stay_length_nights'] = int(nights_match.group(1))
    
    except Exception as e:
        logger.warning(f"Failed to parse filename {filename}: {e}")
    
    return metadata


def extract_number_from_text(text: str, keyword: str) -> Optional[int]:
    """Extract number before keyword from text (e.g., '2 bedrooms' -> 2)."""
    if not text:
        return None
    pattern = rf'(\d+)\s*{keyword}'
    match = re.search(pattern, text.lower())
    return int(match.group(1)) if match else None


def extract_bedroom_count(listing: Dict) -> Optional[int]:
    """Extract bedroom count from structured content."""
    try:
        content = listing.get('structuredContent', {})
        primary_line = content.get('primaryLine', [])
        
        for item in primary_line:
            if item.get('type') == 'BEDINFO':
                body = item.get('body', '')
                # Look for "X bedroom" or "X bedrooms"
                count = extract_number_from_text(body, r'bedroom')
                if count is not None:
                    return count
    except Exception:
        pass
    return None


def extract_bed_count(listing: Dict) -> Optional[int]:
    """Extract bed count from structured content."""
    try:
        content = listing.get('structuredContent', {})
        primary_line = content.get('primaryLine', [])
        
        for item in primary_line:
            if item.get('type') == 'BEDINFO':
                body = item.get('body', '')
                # Look for "X bed" or "X beds" but NOT "bedroom"
                if 'bedroom' not in body.lower():
                    count = extract_number_from_text(body, r'bed')
                    if count is not None:
                        return count
    except Exception:
        pass
    return None


def extract_room_type(title: str) -> str:
    """
    Extract room type from title field.
    
    Examples:
        'Room in El Maarif district' -> 'Private room'
        'Apartment in Casablanca' -> 'Entire home/apt'
        'Condo in downtown' -> 'Entire home/apt'
    """
    if not title:
        return 'Unknown'
    
    title_lower = title.lower()
    
    # Pattern matching for room types
    if 'room in' in title_lower and 'bedroom' not in title_lower:
        return 'Private room'
    elif 'shared room' in title_lower:
        return 'Shared room'
    else:
        return 'Entire home/apt'


def extract_property_type(title: str) -> str:
    """
    Extract property type from title field.
    
    Examples:
        'Apartment in Casablanca' -> 'Apartment'
        'Condo in downtown' -> 'Condominium'
        'Home in Al Hoceima' -> 'House'
    """
    if not title:
        return 'Unknown'
    
    title_lower = title.lower()
    
    # Property type patterns
    property_patterns = {
        'apartment': 'Apartment',
        'condo': 'Condominium',
        'loft': 'Loft',
        'house': 'House',
        'home': 'House',
        'villa': 'Villa',
        'bed and breakfast': 'Bed and breakfast',
        'boutique hotel': 'Boutique hotel',
        'guesthouse': 'Guesthouse',
        'townhouse': 'Townhouse',
        'vacation home': 'House',
    }
    
    for pattern, prop_type in property_patterns.items():
        if pattern in title_lower:
            return prop_type
    
    return 'Other'


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_listing_features(listing: Dict, file_metadata: Dict) -> Dict[str, Any]:
    """
    Extract all features from a single listing JSON object.
    
    Args:
        listing: Raw listing dictionary from JSON
        file_metadata: Metadata parsed from filename
    
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    # Basic identifiers
    features['room_id'] = listing.get('room_id')
    features['listing_name'] = listing.get('name', '')
    features['title'] = listing.get('title', '')
    
    # Location
    features['city_slug'] = file_metadata['city_slug']
    features['city'] = CITY_SLUG_MAP.get(file_metadata['city_slug'], 
                                          file_metadata['city_slug'].replace('_', ' ').title() if file_metadata['city_slug'] else 'Unknown')
    
    # Coordinates
    coords = listing.get('coordinates', {})
    features['latitude'] = coords.get('latitude', np.nan)
    # Handle typo in JSON: both 'longitude' and 'longitud'
    features['longitude'] = coords.get('longitude', coords.get('longitud', np.nan))
    
    # Trip context from filename
    features['season'] = file_metadata['season']
    features['check_in'] = file_metadata['check_in']
    features['stay_length_nights'] = file_metadata['stay_length_nights']
    
    # Calculate check_out
    if features['check_in'] and features['stay_length_nights']:
        features['check_out'] = features['check_in'] + timedelta(days=features['stay_length_nights'])
    else:
        features['check_out'] = None
    
    # Pricing
    price_data = listing.get('price', {})
    unit_price = price_data.get('unit', {})
    
    features['total_price'] = unit_price.get('amount', np.nan)
    features['currency'] = unit_price.get('curency_symbol', 'MAD')  # Note: typo in JSON
    
    # Calculate nightly price
    if features['stay_length_nights'] and features['stay_length_nights'] > 0 and not pd.isna(features['total_price']):
        features['nightly_price'] = round(features['total_price'] / features['stay_length_nights'], 2)
    else:
        features['nightly_price'] = np.nan
    
    # Discount
    discount = unit_price.get('discount', 0)
    if discount and features['total_price']:
        features['discount_rate'] = round(discount / features['total_price'], 4)
    else:
        features['discount_rate'] = 0.0
    
    # Property details
    features['bedroom_count'] = extract_bedroom_count(listing) or 0
    features['bed_count'] = extract_bed_count(listing) or 0
    features['room_type'] = extract_room_type(features['title'])
    features['property_type'] = extract_property_type(features['title'])
    
    # Ratings
    rating = listing.get('rating', {})
    features['rating_value'] = rating.get('value', np.nan)
    
    # Handle reviewCount as string or int
    review_count = rating.get('reviewCount', 0)
    if isinstance(review_count, str):
        review_count = int(review_count) if review_count.isdigit() else 0
    features['rating_count'] = review_count
    
    # Badges
    badges = listing.get('badges', [])
    features['badge_count'] = len(badges)
    features['badges'] = '|'.join(badges) if badges else ''
    features['is_superhost'] = 'SUPERHOST' in badges
    
    # Check passportData for superhost status if not in badges
    passport = listing.get('passportData', {})
    if not features['is_superhost'] and passport:
        features['is_superhost'] = passport.get('isSuperhost', False)
    
    # Images
    images = listing.get('images', [])
    features['image_count'] = len(images)
    
    # File source for traceability
    features['file_source'] = file_metadata['file_source']
    
    return features


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_json_file(filepath: Path) -> List[Dict]:
    """
    Process a single JSON file and extract all listings.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        List of feature dictionaries, one per listing
    """
    try:
        # Parse filename metadata
        file_metadata = parse_filename_metadata(filepath)
        
        # Load JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if not isinstance(data, list):
            logger.warning(f"Expected list in {filepath.name}, got {type(data)}")
            return []
        
        # Extract features from each listing
        listings = []
        for listing in data:
            if not isinstance(listing, dict):
                continue
            
            try:
                features = extract_listing_features(listing, file_metadata)
                listings.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract listing in {filepath.name}: {e}")
                continue
        
        logger.info(f"Processed {filepath.name}: {len(listings)} listings")
        return listings
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return []


def process_all_json_files(input_dir: Path) -> pd.DataFrame:
    """
    Process all JSON files in the input directory.
    
    Args:
        input_dir: Root directory containing JSON files
    
    Returns:
        DataFrame with all extracted listings
    """
    json_files = list(input_dir.rglob('*.json'))
    
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    all_listings = []
    for filepath in json_files:
        listings = process_json_file(filepath)
        all_listings.extend(listings)
    
    if not all_listings:
        logger.error("No listings extracted from any files")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_listings)
    logger.info(f"Extracted {len(df)} total listings from {len(json_files)} files")
    
    return df


def clean_dataset(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """
    Clean and validate the dataset.
    
    Args:
        df: Raw DataFrame
        deduplicate: Whether to remove duplicates
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Starting cleaning: {len(df)} rows")
    
    # Remove rows with missing critical fields
    critical_fields = ['room_id', 'nightly_price', 'city']
    before = len(df)
    df = df.dropna(subset=critical_fields)
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} rows with missing critical fields")
    
    # Remove duplicates
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=['room_id', 'check_in', 'stay_length_nights'], keep='first')
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate listings")
    
    # Remove outliers (prices outside 1st-99th percentile)
    before = len(df)
    p1 = df['nightly_price'].quantile(0.01)
    p99 = df['nightly_price'].quantile(0.99)
    df = df[(df['nightly_price'] >= p1) & (df['nightly_price'] <= p99)]
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} price outliers (< {p1:.2f} or > {p99:.2f} MAD)")
    
    logger.info(f"Cleaning complete: {len(df)} rows remaining")
    
    return df


def generate_summary(df: pd.DataFrame) -> None:
    """Generate and log summary statistics."""
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total listings: {len(df):,}")
    logger.info(f"Total features: {len(df.columns)}")
    logger.info("")
    
    logger.info("Listings by city:")
    for city, count in df['city'].value_counts().items():
        logger.info(f"  {city:20s}: {count:5,}")
    logger.info("")
    
    logger.info("Listings by season:")
    for season, count in df['season'].value_counts().items():
        logger.info(f"  {season:10s}: {count:5,}")
    logger.info("")
    
    logger.info("Price statistics (MAD):")
    logger.info(f"  Mean:   {df['nightly_price'].mean():8.2f}")
    logger.info(f"  Median: {df['nightly_price'].median():8.2f}")
    logger.info(f"  Std:    {df['nightly_price'].std():8.2f}")
    logger.info(f"  Min:    {df['nightly_price'].min():8.2f}")
    logger.info(f"  Max:    {df['nightly_price'].max():8.2f}")
    logger.info("")
    
    logger.info("Room type distribution:")
    for room_type, count in df['room_type'].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {room_type:20s}: {count:5,} ({pct:5.1f}%)")
    logger.info("")
    
    logger.info("Property type distribution:")
    for prop_type, count in df['property_type'].value_counts().head(10).items():
        pct = count / len(df) * 100
        logger.info(f"  {prop_type:20s}: {count:5,} ({pct:5.1f}%)")
    logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Airbnb JSON scrapes to clean CSV dataset'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing JSON files (e.g., data/raw_scrapes_expanded/by_city)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path (e.g., data/morocco_listings_full.csv)'
    )
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Remove duplicate listings based on room_id + check_in + stay_length'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Process all files
    logger.info(f"Processing JSON files from: {args.input_dir}")
    df = process_all_json_files(args.input_dir)
    
    if df.empty:
        logger.error("No data extracted. Exiting.")
        sys.exit(1)
    
    # Clean dataset
    df = clean_dataset(df, deduplicate=args.deduplicate)
    
    # Generate summary
    generate_summary(df)
    
    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved dataset to: {args.output}")
    logger.info(f"File size: {args.output.stat().st_size / 1024**2:.2f} MB")
    logger.info(f"Shape: {df.shape}")


if __name__ == '__main__':
    main()
