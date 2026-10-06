"""Price calculation functions."""


def calculate_discount(price: float, percent: float) -> float:
    """Return the price after applying a percentage discount."""
    if price < 0:
        raise ValueError("price cannot be negative")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return price * (1 - percent)
