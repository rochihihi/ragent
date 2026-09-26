import pytest
from discount import calculate_discount


@pytest.mark.parametrize(
    ("price", "percent", "expected"),
    [(100.0, 10.0, 90.0), (80.0, 0.0, 80.0), (80.0, 100.0, 0.0)],
)
def test_calculate_discount(price: float, percent: float, expected: float) -> None:
    assert calculate_discount(price, percent) == pytest.approx(expected)


def test_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        calculate_discount(-1, 10)
    with pytest.raises(ValueError):
        calculate_discount(100, 101)
