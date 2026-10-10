"""Known overlaps between benchmark datasets and model training data."""


TRAINING_DATASET_OVERLAPS = {
    "yolo26x_obb_text_g1": {"YeniseiGovReports-TD", "school_notebooks_RU"},
    "yolo26s_obb_text_g1": {"YeniseiGovReports-TD", "school_notebooks_RU"},
    "east_50_g1": {"YeniseiGovReports-TD", "ICDAR2015"},
    "east_50_yenisei_gov_reports_g1": {"YeniseiGovReports-TD"},
    "mask2former_line_v0_prev": {"handwritten_essay", "school_notebooks_ru"},
    "trba_base_g1": {
        "school_notebooks_ru",
        "yenisei_gov_reports_hwr",
        "yenisei_gov_reports_prt",
        "cyrillic_handwriting",
    },
    "trba_lite_g1": {
        "school_notebooks_ru",
        "yenisei_gov_reports_hwr",
        "yenisei_gov_reports_prt",
        "cyrillic_handwriting",
    },
    "parseq_s_rukopys": {"school_notebooks_ru", "cyrillic_handwriting"},
    "parseq_b_rukopys": {"school_notebooks_ru", "cyrillic_handwriting"},
    "trocr_large_rukopys_hw": {"cyrillic_handwriting"},
}


def mark_training_overlap(cell: str, model: str, dataset: str) -> str:
    """Append a visible literal star to a known train/test overlap cell."""
    if dataset in TRAINING_DATASET_OVERLAPS.get(model, ()):
        return f"{cell} \\*"
    return cell
