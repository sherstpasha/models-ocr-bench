"""Ranking helpers shared by benchmark summaries."""


def mean_metric_ranks(metric_maps, higher_is_better=None):
    """Average conventional ranks, where the best metric value receives rank 1."""
    if higher_is_better is None:
        higher_is_better = [True] * len(metric_maps)
    if len(higher_is_better) != len(metric_maps):
        raise ValueError("higher_is_better must match metric_maps length")

    per_metric = []
    for metric, descending in zip(metric_maps, higher_is_better):
        available = sorted(
            ((value, model) for model, value in metric.items() if value is not None),
            reverse=descending,
        )
        ranks = {}
        index = 0
        while index < len(available):
            end = index + 1
            while end < len(available) and available[end][0] == available[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2
            for _value, model in available[index:end]:
                ranks[model] = average_rank
            index = end
        per_metric.append(ranks)

    models = set().union(*(metric.keys() for metric in metric_maps))
    return {
        model: (
            sum(ranks[model] for ranks in per_metric) / len(per_metric)
            if all(model in ranks for ranks in per_metric)
            else None
        )
        for model in models
    }
