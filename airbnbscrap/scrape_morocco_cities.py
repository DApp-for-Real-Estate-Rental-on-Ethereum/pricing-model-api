#!/usr/bin/env python3
"""
Multi-City Multi-Season Airbnb Scraper for Morocco

This script orchestrates large-scale data collection across multiple Moroccan cities
and seasonal periods to build a robust pricing model dataset.

Features:
- Scrapes 10+ major Moroccan cities
- Covers all 4 seasons (Winter, Spring, Summer, Fall)
- Multiple stay durations (weekend, week, 2-weeks)
- Automatic retry logic and rate limiting
- Progress tracking and error logging
- Organized output by city and season

Usage:
    python scrape_morocco_cities.py --all-cities --all-seasons --output-dir data/raw_scrapes
    python scrape_morocco_cities.py --cities marrakech rabat --seasons summer winter
    python scrape_morocco_cities.py --cities casablanca --quick-test  # Test mode
"""

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CITY DEFINITIONS - Major Moroccan Cities with Verified Bounding Boxes
# Coordinates verified against latlong.net, geodatos.net, latitude.to
# ============================================================================

MOROCCO_CITIES = {
    'casablanca': {
        'name': 'Casablanca',
        'ne_lat': 33.6318,
        'ne_long': -7.5086,
        'sw_lat': 33.5170,
        'sw_long': -7.6890,
        'center_lat': 33.59,
        'center_long': -7.61,
        'zoom': 12,
        'description': 'Economic capital, largest city (~3.7M pop.)',
        'verified': True
    },
    'marrakech': {
        'name': 'Marrakech',
        'ne_lat': 31.6695,
        'ne_long': -7.9400,
        'sw_lat': 31.5900,
        'sw_long': -8.0600,
        'center_lat': 31.63,
        'center_long': -8.00,
        'zoom': 12,
        'description': 'Tourist hub, medina & palaces (UNESCO site)',
        'verified': True
    },
    'rabat': {
        'name': 'Rabat',
        'ne_lat': 34.0531,
        'ne_long': -6.7787,
        'sw_lat': 33.9500,
        'sw_long': -6.8900,
        'center_lat': 34.02,
        'center_long': -6.84,
        'zoom': 12,
        'description': 'Capital city, administrative center',
        'verified': True
    },
    'fes': {
        'name': 'Fes',
        'ne_lat': 34.0700,
        'ne_long': -4.9700,
        'sw_lat': 34.0200,
        'sw_long': -5.0300,
        'center_lat': 34.03,
        'center_long': -5.00,
        'zoom': 12,
        'description': 'Cultural capital, ancient medina (UNESCO site)',
        'verified': True
    },
    'tangier': {
        'name': 'Tangier',
        'ne_lat': 35.8100,
        'ne_long': -5.7600,
        'sw_lat': 35.7400,
        'sw_long': -5.8400,
        'center_lat': 35.77,
        'center_long': -5.80,
        'zoom': 12,
        'description': 'Gateway to Europe, coastal city (ferry port)',
        'verified': True
    },
    'agadir': {
        'name': 'Agadir',
        'ne_lat': 30.4500,
        'ne_long': -9.5600,
        'sw_lat': 30.3900,
        'sw_long': -9.6400,
        'center_lat': 30.43,
        'center_long': -9.60,
        'zoom': 12,
        'description': 'Beach resort city, Atlantic coast (post-1960 rebuild)',
        'verified': True
    },
    'meknes': {
        'name': 'Meknes',
        'ne_lat': 33.9200,
        'ne_long': -5.5200,
        'sw_lat': 33.8700,
        'sw_long': -5.5800,
        'center_lat': 33.89,
        'center_long': -5.55,
        'zoom': 12,
        'description': 'Imperial city, historical capital',
        'verified': True
    },
    'oujda': {
        'name': 'Oujda',
        'ne_lat': 34.7000,
        'ne_long': -1.8800,
        'sw_lat': 34.6600,
        'sw_long': -1.9400,
        'center_lat': 34.69,
        'center_long': -1.91,
        'zoom': 12,
        'description': 'Eastern border city, trade hub',
        'verified': True
    },
    'tetouan': {
        'name': 'Tetouan',
        'ne_lat': 35.5900,
        'ne_long': -5.3500,
        'sw_lat': 35.5500,
        'sw_long': -5.3900,
        'center_lat': 35.57,
        'center_long': -5.37,
        'zoom': 12,
        'description': 'Spanish-influenced medina (UNESCO site)',
        'verified': True
    },
    'essaouira': {
        'name': 'Essaouira',
        'ne_lat': 31.5300,
        'ne_long': -9.7500,
        'sw_lat': 31.4900,
        'sw_long': -9.7900,
        'center_lat': 31.51,
        'center_long': -9.77,
        'zoom': 13,
        'description': 'Windsurfing destination, UNESCO port',
        'verified': True
    },
    'chefchaouen': {
        'name': 'Chefchaouen',
        'ne_lat': 35.1800,
        'ne_long': -5.2500,
        'sw_lat': 35.1600,
        'sw_long': -5.2800,
        'center_lat': 35.17,
        'center_long': -5.27,
        'zoom': 14,
        'description': 'Blue city, mountain town',
        'verified': True
    },
    'ouarzazate': {
        'name': 'Ouarzazate',
        'ne_lat': 30.9400,
        'ne_long': -6.8800,
        'sw_lat': 30.9000,
        'sw_long': -6.9400,
        'center_lat': 30.92,
        'center_long': -6.90,
        'zoom': 13,
        'description': 'Desert gateway, film studios (Hollywood of Morocco)',
        'verified': True
    },
    'al_hociema': {
        'name': 'Al Hoceima',
        'ne_lat': 35.2650,
        'ne_long': -3.9200,
        'sw_lat': 35.2250,
        'sw_long': -3.9700,
        'zoom': 13,
        'description': 'Mediterranean coastal city, Rif mountains, popular summer resort'
    }
}


# ============================================================================
# SEASONAL DATE RANGES (2025-2026)
# ============================================================================

SEASONS = {
    'winter_2025': {
        'name': 'Winter 2025',
        'periods': [
            ('2025-12-15', '2025-12-18'),  # Weekend
            ('2025-12-20', '2025-12-27'),  # Week (Christmas)
            ('2026-01-05', '2026-01-12'),  # Week
            ('2026-01-15', '2026-01-29'),  # 2 weeks
            ('2026-02-01', '2026-02-08'),  # Week
        ]
    },
    'spring_2025': {
        'name': 'Spring 2025',
        'periods': [
            ('2025-03-07', '2025-03-10'),  # Weekend
            ('2025-03-15', '2025-03-22'),  # Week
            ('2025-04-05', '2025-04-12'),  # Week
            ('2025-04-15', '2025-04-29'),  # 2 weeks (Easter period)
            ('2025-05-01', '2025-05-08'),  # Week
            ('2025-05-20', '2025-05-27'),  # Week
        ]
    },
    'summer_2025': {
        'name': 'Summer 2025',
        'periods': [
            ('2025-06-07', '2025-06-10'),  # Weekend
            ('2025-06-15', '2025-06-22'),  # Week
            ('2025-07-01', '2025-07-08'),  # Week
            ('2025-07-10', '2025-07-24'),  # 2 weeks (peak summer)
            ('2025-08-01', '2025-08-08'),  # Week
            ('2025-08-15', '2025-08-29'),  # 2 weeks (peak summer)
        ]
    },
    'fall_2025': {
        'name': 'Fall 2025',
        'periods': [
            ('2025-09-05', '2025-09-08'),  # Weekend
            ('2025-09-15', '2025-09-22'),  # Week
            ('2025-10-01', '2025-10-08'),  # Week
            ('2025-10-15', '2025-10-29'),  # 2 weeks
            ('2025-11-01', '2025-11-08'),  # Week
            ('2025-11-20', '2025-11-27'),  # Week
        ]
    }
}


# ============================================================================
# SCRAPING ORCHESTRATION
# ============================================================================

class MoroccoScraper:
    """Orchestrates multi-city, multi-season Airbnb data collection."""
    
    def __init__(self, output_dir: Path, currency: str = 'MAD', language: str = 'en', 
                 delay: int = 3, proxy_url: str = ''):
        self.output_dir = Path(output_dir)
        self.currency = currency
        self.language = language
        self.delay = delay  # Seconds between requests
        self.proxy_url = proxy_url
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'total_listings': 0
        }
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'by_city').mkdir(exist_ok=True)
        (self.output_dir / 'by_season').mkdir(exist_ok=True)
        
    def scrape_city_season(self, city_key: str, season_key: str, 
                          check_in: str, check_out: str) -> Dict:
        """Scrape a single city for a specific date range."""
        city = MOROCCO_CITIES[city_key]
        
        logger.info(f"Scraping {city['name']} ({check_in} to {check_out})")
        
        # Build output filename
        nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
        filename = f"{city_key}_{season_key}_{check_in}_{nights}nights.json"
        output_path = self.output_dir / 'by_city' / city_key / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if already scraped
        if output_path.exists():
            logger.info(f"✓ Already scraped: {filename}")
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                    return {
                        'status': 'cached',
                        'listings_count': len(existing_data),
                        'file': str(output_path)
                    }
            except Exception as e:
                logger.warning(f"Cached file corrupted, re-scraping: {e}")
        
        # Build command for scrapingairbnb.py
        cmd = [
            'python', 'scrapingairbnb.py', 'search',
            '--check-in', check_in,
            '--check-out', check_out,
            '--ne-lat', str(city['ne_lat']),
            '--ne-long', str(city['ne_long']),
            '--sw-lat', str(city['sw_lat']),
            '--sw-long', str(city['sw_long']),
            '--zoom', str(city['zoom']),
            '--currency', self.currency,
            '--language', self.language,
            '--output', str(output_path)
        ]
        
        if self.proxy_url:
            cmd.extend(['--proxy-url', self.proxy_url])
        
        # Execute scraping
        try:
            self.stats['total_requests'] += 1
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                # Count listings
                with open(output_path, 'r') as f:
                    data = json.load(f)
                    listings_count = len(data)
                
                self.stats['successful'] += 1
                self.stats['total_listings'] += listings_count
                
                logger.info(f"✅ Success: {listings_count} listings → {filename}")
                
                # Rate limiting
                time.sleep(self.delay)
                
                return {
                    'status': 'success',
                    'listings_count': listings_count,
                    'file': str(output_path)
                }
            else:
                self.stats['failed'] += 1
                logger.error(f"❌ Failed: {result.stderr}")
                return {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            self.stats['failed'] += 1
            logger.error(f"⏱️  Timeout: {filename}")
            return {'status': 'timeout'}
        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"💥 Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def scrape_all(self, cities: List[str], seasons: List[str]) -> Dict:
        """Scrape all combinations of cities and seasons."""
        results = []
        total_tasks = sum(
            len(SEASONS[s]['periods']) 
            for s in seasons if s in SEASONS
        ) * len(cities)
        
        logger.info(f"🚀 Starting scraping campaign: {len(cities)} cities × {len(seasons)} seasons")
        logger.info(f"📊 Total scraping tasks: {total_tasks}")
        logger.info("=" * 70)
        
        task_num = 0
        
        for city_key in cities:
            if city_key not in MOROCCO_CITIES:
                logger.warning(f"⚠️  Unknown city: {city_key}")
                continue
                
            city = MOROCCO_CITIES[city_key]
            logger.info(f"\n🏙️  City: {city['name']} - {city['description']}")
            
            for season_key in seasons:
                if season_key not in SEASONS:
                    logger.warning(f"⚠️  Unknown season: {season_key}")
                    continue
                
                season = SEASONS[season_key]
                logger.info(f"  📅 Season: {season['name']}")
                
                for check_in, check_out in season['periods']:
                    task_num += 1
                    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
                    
                    logger.info(f"    [{task_num}/{total_tasks}] {check_in} → {check_out} ({nights} nights)")
                    
                    result = self.scrape_city_season(
                        city_key, season_key, check_in, check_out
                    )
                    
                    results.append({
                        'city': city_key,
                        'season': season_key,
                        'check_in': check_in,
                        'check_out': check_out,
                        'nights': nights,
                        **result
                    })
        
        # Save summary
        self._save_summary(results)
        
        return {
            'results': results,
            'stats': self.stats
        }
    
    def _save_summary(self, results: List[Dict]) -> None:
        """Save scraping campaign summary."""
        summary_path = self.output_dir / 'scraping_summary.json'
        
        summary = {
            'campaign_stats': self.stats,
            'timestamp': date.today().isoformat(),
            'currency': self.currency,
            'language': self.language,
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 SCRAPING CAMPAIGN COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total requests: {self.stats['total_requests']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total listings collected: {self.stats['total_listings']:,}")
        logger.info(f"Summary saved: {summary_path}")
        logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-city, multi-season Airbnb scraper for Morocco',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all cities and seasons (full dataset)
  python scrape_morocco_cities.py --all-cities --all-seasons
  
  # Scrape specific cities for summer
  python scrape_morocco_cities.py --cities marrakech casablanca agadir --seasons summer_2025
  
  # Quick test with one city
  python scrape_morocco_cities.py --cities rabat --seasons winter_2025 --quick-test
  
  # Use proxy
  python scrape_morocco_cities.py --all-cities --all-seasons --proxy-url http://proxy:8080
        """
    )
    
    parser.add_argument(
        '--cities',
        nargs='+',
        choices=list(MOROCCO_CITIES.keys()),
        help='Cities to scrape (space-separated)'
    )
    parser.add_argument(
        '--all-cities',
        action='store_true',
        help='Scrape all available cities'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        choices=list(SEASONS.keys()),
        help='Seasons to scrape (space-separated)'
    )
    parser.add_argument(
        '--all-seasons',
        action='store_true',
        help='Scrape all seasons'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../data/raw_scrapes_expanded'),
        help='Output directory for scraped data'
    )
    parser.add_argument(
        '--currency',
        default='MAD',
        help='Currency code (default: MAD)'
    )
    parser.add_argument(
        '--language',
        default='en',
        help='Language code (default: en)'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=3,
        help='Delay between requests in seconds (default: 3)'
    )
    parser.add_argument(
        '--proxy-url',
        default='',
        help='Proxy URL for requests'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Test mode: only scrape first period of each season'
    )
    parser.add_argument(
        '--list-cities',
        action='store_true',
        help='List all available cities and exit'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List cities and exit
    if args.list_cities:
        print("\n🏙️  Available Moroccan Cities:")
        print("=" * 70)
        for key, city in MOROCCO_CITIES.items():
            print(f"{key:15s} - {city['name']:15s} | {city['description']}")
        print(f"\nTotal: {len(MOROCCO_CITIES)} cities")
        return
    
    # Determine cities to scrape
    if args.all_cities:
        cities = list(MOROCCO_CITIES.keys())
    elif args.cities:
        cities = args.cities
    else:
        print("❌ Error: Specify --cities or --all-cities")
        sys.exit(1)
    
    # Determine seasons to scrape
    if args.all_seasons:
        seasons = list(SEASONS.keys())
    elif args.seasons:
        seasons = args.seasons
    else:
        print("❌ Error: Specify --seasons or --all-seasons")
        sys.exit(1)
    
    # Quick test mode
    if args.quick_test:
        logger.info("🧪 QUICK TEST MODE: Only first period per season")
        # Limit to first period
        for season_key in SEASONS:
            SEASONS[season_key]['periods'] = SEASONS[season_key]['periods'][:1]
    
    # Initialize scraper
    scraper = MoroccoScraper(
        output_dir=args.output_dir,
        currency=args.currency,
        language=args.language,
        delay=args.delay,
        proxy_url=args.proxy_url
    )
    
    # Run scraping campaign
    scraper.scrape_all(cities, seasons)


if __name__ == '__main__':
    main()
