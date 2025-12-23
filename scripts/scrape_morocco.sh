#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/scrape_morocco.sh [CHECK_IN] [CHECK_OUT]
# Example:
#   bash scripts/scrape_morocco.sh 2025-03-01 2025-03-05
#
# Environment overrides:
#   PYTHON         - Python executable (default: python)
#   CURRENCY       - Currency code (default: MAD)
#   LANGUAGE       - Language code (default: fr)
#   PRICE_MIN/MAX  - Integer price filters (default: 0 = no filter)
#   AMENITIES      - Optional comma-separated amenity ids (e.g., "4,8,30")
#   OUTPUT_DIR     - Base folder for JSON outputs (default: data/raw)

PYTHON_BIN=${PYTHON:-python}
CHECK_IN=${1:-2025-03-01}
CHECK_OUT=${2:-2025-03-05}
CURRENCY=${CURRENCY:-MAD}
LANGUAGE=${LANGUAGE:-fr}
PRICE_MIN=${PRICE_MIN:-0}
PRICE_MAX=${PRICE_MAX:-0}
AMENITIES=${AMENITIES:-}
OUTPUT_DIR=${OUTPUT_DIR:-data/raw}

if [[ -z $CHECK_IN || -z $CHECK_OUT ]]; then
  echo "Usage: $0 [CHECK_IN] [CHECK_OUT]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Bounding boxes gathered from manual inspection of Airbnb map for key Moroccan cities.
# Add more cities by extending the arrays below.
declare -A NE_LAT=(
  [casablanca]=33.632
  [marrakech]=31.690
  [rabat]=34.053
  [tangier]=35.812
  [agadir]=30.462
  [fes]=34.069
)

declare -A NE_LONG=(
  [casablanca]=-7.477
  [marrakech]=-7.932
  [rabat]=-6.770
  [tangier]=-5.742
  [agadir]=-9.508
  [fes]=-4.956
)

declare -A SW_LAT=(
  [casablanca]=33.518
  [marrakech]=31.526
  [rabat]=33.935
  [tangier]=35.703
  [agadir]=30.376
  [fes]=34.000
)

declare -A SW_LONG=(
  [casablanca]=-7.714
  [marrakech]=-8.103
  [rabat]=-6.936
  [tangier]=-5.931
  [agadir]=-9.635
  [fes]=-5.080
)

cities=(casablanca marrakech rabat tangier agadir fes)

for city in "${cities[@]}"; do
  echo "➜ Scraping $city between $CHECK_IN and $CHECK_OUT..."
  outfile="$OUTPUT_DIR/${city}_${CHECK_IN}_${CHECK_OUT}.json"
  $PYTHON_BIN scrapingairbnb.py search \
    --check-in "$CHECK_IN" \
    --check-out "$CHECK_OUT" \
    --ne-lat "${NE_LAT[$city]}" \
    --ne-long "${NE_LONG[$city]}" \
    --sw-lat "${SW_LAT[$city]}" \
    --sw-long "${SW_LONG[$city]}" \
    --currency "$CURRENCY" \
    --language "$LANGUAGE" \
    --price-min "$PRICE_MIN" \
    --price-max "$PRICE_MAX" \
    --output "$outfile" \
    ${AMENITIES:+--amenities "$AMENITIES"}

done

echo "✅ Finished scraping ${#cities[@]} cities. Outputs saved under $OUTPUT_DIR"
