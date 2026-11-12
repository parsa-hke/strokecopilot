"""Small statistics shared by the Vision B checks (EXP-16, EXP-17). Pure Python, so a Kaggle notebook can import it
without the ISLES'24 dependencies."""

import math

Z95 = 1.959964  # as in operating_point.py and isles_check.py, so recorded intervals reproduce


def wilson(k, n, z=Z95):
    """Wilson score interval for a proportion k/n. (None, None) when n is 0."""
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
