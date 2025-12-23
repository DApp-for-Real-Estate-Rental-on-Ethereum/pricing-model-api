#!/usr/bin/env python3
"""Utility helpers around the `pyairbnb` SDK.

This script provides a tiny command-line interface for the most common
scraping flows exposed by `pyairbnb` so you can quickly fetch listing
search results, listing details, reviews, and calendars without writing
extra boilerplate each time.

Usage examples (after `pip install pyairbnb`):

	# Bounding-box search for Quito with Wi-Fi (amenity id 4)
	python scrapingairbnb.py search \
		--check-in 2025-10-01 --check-out 2025-10-04 \
		--ne-lat -0.6747 --ne-long -90.3005 \
		--sw-lat -0.7596 --sw-long -90.3672 \
		--currency MXN --amenities 4 --output search.json

	# Fetch details & availability for a single listing URL
	python scrapingairbnb.py details --room-url https://www.airbnb.com/rooms/30931885 \
		--check-in 2025-10-12 --check-out 2025-10-17 --currency USD \
		--language en --adults 2 --output details.json

	# Pull reviews for the same listing
	python scrapingairbnb.py reviews --room-url https://www.airbnb.com/rooms/30931885 \
		--language fr --output reviews.json

All sub-commands accept an optional ``--proxy-url`` argument in case you need
to tunnel traffic through a proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pyairbnb


class PyAirbnbDataError(SystemExit):
	"""Raised when pyairbnb returns malformed or empty data."""


def _raise_data_error(message: str, exc: Exception) -> None:
	"""Convert library parsing issues into actionable CLI errors."""
	raise PyAirbnbDataError(
		f"{message}\nUnderlying error: {exc.__class__.__name__}: {exc}"
	) from exc


def _ensure_iso_date(value: str) -> str:
	"""Validate ISO-8601 date strings because Airbnb rejects other formats."""

	try:
		# `fromisoformat` raises ValueError for bad inputs; we only need validation.
		from datetime import date

		date.fromisoformat(value)
	except ValueError as exc:  # pragma: no cover - defensive guard
		raise argparse.ArgumentTypeError(
			f"{value!r} is not a valid ISO date (expected YYYY-MM-DD)"
		) from exc
	return value


def _amenities_list(value: str) -> List[int]:
	if not value:
		return []
	try:
		return [int(v.strip()) for v in value.split(",") if v.strip()]
	except ValueError as exc:  # pragma: no cover - defensive guard
		raise argparse.ArgumentTypeError(
			"Amenities must be a comma-separated list of integers"
		) from exc


def _dump_json(data, output: Path) -> None:
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_search(args: argparse.Namespace) -> None:
	amenities = args.amenities or []
	payload = pyairbnb.search_all(
		check_in=args.check_in,
		check_out=args.check_out,
		ne_lat=args.ne_lat,
		ne_long=args.ne_long,
		sw_lat=args.sw_lat,
		sw_long=args.sw_long,
		zoom_value=args.zoom,
		price_min=args.price_min,
		price_max=args.price_max,
		place_type=args.place_type,
		amenities=amenities,
		free_cancellation=args.free_cancellation,
		currency=args.currency,
		language=args.language,
		proxy_url=args.proxy_url,
	)
	_dump_json(payload, args.output)
	print(
		f"Saved {len(payload)} listings to {args.output}. "
		"Use jq or pandas for deeper analysis."
	)


def cmd_url_search(args: argparse.Namespace) -> None:
	hash_value = args.hash
	if not hash_value:
		hash_value = pyairbnb.fetch_stays_search_hash()
	payload = pyairbnb.search_all_from_url(
		args.url,
		currency=args.currency,
		language=args.language,
		proxy_url=args.proxy_url,
		hash=hash_value,
	)
	_dump_json(payload, args.output)
	print(f"Saved {len(payload)} URL search listings to {args.output}.")


def cmd_details(args: argparse.Namespace) -> None:
	if not args.room_url and not args.room_id:
		raise SystemExit("You must provide either --room-url or --room-id for details.")
	try:
		payload = pyairbnb.get_details(
			room_url=args.room_url,
			room_id=args.room_id,
			currency=args.currency,
			adults=args.adults,
			language=args.language,
			proxy_url=args.proxy_url,
			check_in=args.check_in,
			check_out=args.check_out,
		)
	except (KeyError, TypeError) as exc:
		_prompt = (
			"pyairbnb couldn't load that listing. Double-check that the room URL/ID is valid,"
			" public, and not geo-restricted."
		)
		_raise_data_error(_prompt, exc)
	_dump_json(payload, args.output)
	print(f"Saved listing details to {args.output}.")


def cmd_reviews(args: argparse.Namespace) -> None:
	try:
		payload = pyairbnb.get_reviews(args.room_url, args.language, args.proxy_url)
	except (KeyError, TypeError) as exc:
		_raise_data_error(
			"Failed to parse reviews. The listing might be unavailable or blocked in your region.",
			exc,
		)
	_dump_json(payload, args.output)
	print(f"Saved {len(payload)} reviews to {args.output}.")


def cmd_calendar(args: argparse.Namespace) -> None:
	payload = pyairbnb.get_calendar(args.api_key or "", args.room_id, args.proxy_url)
	_dump_json(payload, args.output)
	print(f"Saved calendar data for room {args.room_id} to {args.output}.")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Minimal command-line wrapper around pyairbnb for scraping tasks."
	)
	shared = argparse.ArgumentParser(add_help=False)
	shared.add_argument(
		"--proxy-url",
		default="",
		help="Optional HTTPS proxy URL passed through to pyairbnb's HTTP requests.",
	)
	sub = parser.add_subparsers(dest="command", required=True)

	search = sub.add_parser(
		"search", help="Search listings via bounding box coordinates.", parents=[shared]
	)
	search.add_argument("--check-in", type=_ensure_iso_date, required=True)
	search.add_argument("--check-out", type=_ensure_iso_date, required=True)
	search.add_argument("--ne-lat", type=float, required=True, help="North-east latitude")
	search.add_argument("--ne-long", type=float, required=True, help="North-east longitude")
	search.add_argument("--sw-lat", type=float, required=True, help="South-west latitude")
	search.add_argument("--sw-long", type=float, required=True, help="South-west longitude")
	search.add_argument("--zoom", type=int, default=10, help="Map zoom level from Airbnb UI")
	search.add_argument("--price-min", type=int, default=0)
	search.add_argument("--price-max", type=int, default=0)
	search.add_argument(
		"--place-type",
		default="",
		help="Optional room type filter (Private room / Entire home/apt / etc).",
	)
	search.add_argument(
		"--amenities",
		type=_amenities_list,
		default=[],
		help="Comma-separated amenity IDs (see Airbnb amenity map).",
	)
	search.add_argument(
		"--free-cancellation",
		action="store_true",
		help="Filter for listings with flexible/free cancellation.",
	)
	search.add_argument("--currency", default="USD")
	search.add_argument("--language", default="en")
	search.add_argument(
		"--output",
		type=Path,
		default=Path("search_results.json"),
		help="Where to save the JSON payload.",
	)
	search.set_defaults(func=cmd_search)

	url_search = sub.add_parser(
		"url-search", help="Search using a full Airbnb map URL.", parents=[shared]
	)
	url_search.add_argument("url", help="Airbnb search URL with supported query params.")
	url_search.add_argument("--currency", default="USD")
	url_search.add_argument("--language", default="en")
	url_search.add_argument("--hash", default="", help="Optional stays search hash.")
	url_search.add_argument(
		"--output",
		type=Path,
		default=Path("search_from_url.json"),
		help="Where to save the JSON payload.",
	)
	url_search.set_defaults(func=cmd_url_search)

	details = sub.add_parser(
		"details", help="Fetch detailed metadata for a listing.", parents=[shared]
	)
	details.add_argument("--room-url", help="Full Airbnb room URL.")
	details.add_argument("--room-id", help="Room ID if you prefer numeric lookup.")
	details.add_argument("--currency", default="USD")
	details.add_argument("--language", default="en")
	details.add_argument("--adults", type=int, default=2)
	details.add_argument("--check-in", type=_ensure_iso_date)
	details.add_argument("--check-out", type=_ensure_iso_date)
	details.add_argument(
		"--output",
		type=Path,
		default=Path("details.json"),
		help="Where to save the JSON payload.",
	)
	details.set_defaults(func=cmd_details)

	reviews = sub.add_parser(
		"reviews", help="Retrieve reviews for a listing URL.", parents=[shared]
	)
	reviews.add_argument("--room-url", required=True)
	reviews.add_argument("--language", default="en")
	reviews.add_argument(
		"--output",
		type=Path,
		default=Path("reviews.json"),
		help="Where to save the JSON payload.",
	)
	reviews.set_defaults(func=cmd_reviews)

	calendar = sub.add_parser(
		"calendar", help="Fetch availability calendar for a room id.", parents=[shared]
	)
	calendar.add_argument("--room-id", required=True)
	calendar.add_argument("--api-key", default="", help="Optional API key if you already have one.")
	calendar.add_argument(
		"--output",
		type=Path,
		default=Path("calendar.json"),
		help="Where to save the JSON payload.",
	)
	calendar.set_defaults(func=cmd_calendar)

	return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)


if __name__ == "__main__":
	main()

