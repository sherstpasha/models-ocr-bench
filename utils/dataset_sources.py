"""Dataset source links used by benchmark summaries."""


def dataset_source_url(config):
    if config.get("source_url"):
        return config["source_url"]
    if config.get("repository"):
        return f"https://huggingface.co/datasets/{config['repository']}"
    if config.get("kaggle_dataset"):
        return f"https://www.kaggle.com/datasets/{config['kaggle_dataset']}"
    return None


def markdown_dataset_header(name, config):
    source = dataset_source_url(config)
    return f"[{name}]({source})" if source else name
