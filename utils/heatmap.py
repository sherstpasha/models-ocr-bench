"""Muted text-only heatmap helpers for Markdown benchmark tables."""


def heat_color(value, low, high, higher_is_better=True):
    score = 1.0 if high == low else (value - low) / (high - low)
    if not higher_is_better:
        score = 1.0 - score
    bad = (179, 92, 92)
    middle = (138, 122, 82)
    good = (79, 138, 91)
    start, end, mix = (
        (bad, middle, score * 2)
        if score <= 0.5
        else (middle, good, (score - 0.5) * 2)
    )
    red, green, blue = (
        round(start[index] + (end[index] - start[index]) * mix)
        for index in range(3)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def heat_span(text, value, value_range, higher_is_better=True, is_best=False):
    low, high = value_range
    content = f"<b>{text}</b>" if is_best else text
    return (
        f'<span style="color:{heat_color(value, low, high, higher_is_better)}">'
        f"{content}</span>"
    )
