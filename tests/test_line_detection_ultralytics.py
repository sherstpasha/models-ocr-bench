from types import SimpleNamespace

import torch

from scripts.line_detection.benchmark_ultralytics import detect


class FakeModel:
    def predict(self, *args, **kwargs):
        boxes = SimpleNamespace(
            cls=torch.tensor([0.0]),
            xyxy=torch.tensor([[10.0, 20.0, 110.0, 45.0]]),
        )
        return [SimpleNamespace(boxes=boxes, masks=None, names={0: "textline"})]


def test_detection_only_checkpoint_boxes_become_line_polygons():
    polygons = detect(
        FakeModel(),
        "unused.jpg",
        {
            "imgsz": 640,
            "confidence": 0.25,
            "device": "cpu",
        },
    )

    assert polygons == [
        [[[10.0, 20.0], [110.0, 20.0], [110.0, 45.0], [10.0, 45.0]]]
    ]
