"""Read-only sales CSV analysis with an argparse command-line interface."""

import argparse
import csv
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path


def parse_date(value):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date does not exist") from exc


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def summarize(path, start=None, end=None, top=3):
    """Validate all rows, then aggregate completed orders within the date range."""
    if start and end and start > end:
        raise ValueError("start date must not be later than end date")
    if top < 1:
        raise ValueError("top must be positive")
    products, days, seen = {}, {}, set()
    orders_in_range = excluded = orders = units = 0
    revenue = Decimal("0")
    required = {"order_id", "date", "product", "quantity", "unit_price", "status"}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError("CSV needs unique headers: " + ", ".join(sorted(required)))
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"line {line}: column count does not match header")
            row = {key: value.strip() for key, value in row.items()}
            if not row["order_id"] or row["order_id"] in seen:
                raise ValueError(f"line {line}: missing or duplicate order_id")
            seen.add(row["order_id"])
            try:
                day = parse_date(row["date"])
            except argparse.ArgumentTypeError as exc:
                raise ValueError(f"line {line}: {exc}") from exc
            if not re.fullmatch(r"[0-9]{1,7}", row["quantity"]):
                raise ValueError(f"line {line}: quantity must be an integer from 1 to 1000000")
            quantity = int(row["quantity"])
            if not 1 <= quantity <= 1000000:
                raise ValueError(f"line {line}: quantity must be from 1 to 1000000")
            if not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,2})?", row["unit_price"]):
                raise ValueError(f"line {line}: invalid unit_price")
            if not row["product"] or row["status"] not in {"completed", "cancelled", "refunded"}:
                raise ValueError(f"line {line}: invalid product or status")
            if (start and day < start) or (end and day > end):
                continue
            orders_in_range += 1
            if row["status"] != "completed":
                excluded += 1
                continue
            amount = Decimal(row["unit_price"]) * quantity
            orders += 1
            units += quantity
            revenue += amount
            for groups, key in ((products, row["product"]), (days, day.isoformat())):
                group = groups.setdefault(key, {"orders": 0, "units": 0, "revenue": Decimal("0")})
                group["orders"] += 1
                group["units"] += quantity
                group["revenue"] += amount

    def output_group(key_name, key, group):
        return {key_name: key, **group, "revenue": format(group["revenue"], ".2f")}

    ranked = sorted(products.items(), key=lambda pair: (-pair[1]["revenue"], pair[0]))[:top]
    return {
        "input": str(Path(path).resolve()),
        "currency": "CNY",
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "orders_in_range": orders_in_range,
        "excluded_orders": excluded,
        "completed_orders": orders,
        "units": units,
        "revenue": format(revenue, ".2f"),
        "top_products": [output_group("product", key, value) for key, value in ranked],
        "daily": [output_group("date", key, value) for key, value in sorted(days.items())],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze completed sales orders in a CSV without modifying any files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python analyze.py --input sales.csv --start 2026-10-01 --top 2. "
            "Required columns: order_id,date,product,quantity,unit_price,status. "
            "Output: JSON on stdout; errors: stderr and exit code 2."
        ),
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "assets" / "sample-sales.csv",
        help="UTF-8 sales CSV path; defaults to the bundled demo, relative to this script",
    )
    parser.add_argument("--start", type=parse_date, help="inclusive first day, YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, help="inclusive last day, YYYY-MM-DD")
    parser.add_argument(
        "--top", type=positive_int, default=3, help="number of products ranked by revenue"
    )
    args = parser.parse_args(argv)
    try:
        report = summarize(args.input, args.start, args.end, args.top)
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
