from pathlib import Path


def prediction_key(image_path, dataset_config):
    """Return the COCO file_name corresponding to an image on disk."""
    prefix = dataset_config.get("filename_prefix")
    if not prefix:
        return Path(image_path).name

    relative = Path(image_path).relative_to(dataset_config["folder"])
    return prefix + "_".join(relative.parts)


def result_is_compatible(stats, dataset_config):
    """Reject results produced before dataset-specific filename mapping."""
    if not stats:
        return False
    if not dataset_config.get("filename_prefix"):
        return True

    metrics = stats.get("accuracy_metrics", {})
    return metrics.get("evaluated_images") == stats.get("num_images")
