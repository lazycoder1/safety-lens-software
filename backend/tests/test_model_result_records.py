import model_manager


class _BulkRows:
    def __init__(self, rows):
        self.rows = rows
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return self.rows


class _Boxes:
    def __init__(self, rows):
        self.data = _BulkRows(rows)

    def __iter__(self):
        raise AssertionError("box tensors must not be iterated scalar-by-scalar")


class _Result:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _Keypoints:
    def __init__(self, rows):
        self.data = _BulkRows(rows)


def test_result_records_use_one_bulk_host_transfer_and_preserve_semantics():
    boxes = _Boxes(
        [
            [10.9, -2.9, 30.1, 40.99, 0.8125, 7.0],
            [1.2, 2.3, 99.8, 100.7, 0.333, 2.0],
        ]
    )

    records = model_manager._records_from_results([_Result(boxes)])

    assert boxes.data.tolist_calls == 1
    assert records == [
        {
            "class_id": 7,
            "confidence": 0.8125,
            "bbox": [10, -2, 30, 40],
        },
        {
            "class_id": 2,
            "confidence": 0.333,
            "bbox": [1, 2, 99, 100],
        },
    ]


def test_result_records_support_tracked_box_rows():
    boxes = _Boxes(
        [
            # Tracked boxes insert track_id before confidence and class.
            [1.0, 2.0, 3.0, 4.0, 987.0, 0.75, 5.0],
        ]
    )

    assert model_manager._records_from_results([_Result(boxes)]) == [
        {
            "class_id": 5,
            "confidence": 0.75,
            "bbox": [1, 2, 3, 4],
        }
    ]
    assert boxes.data.tolist_calls == 1


def test_result_records_preserve_pose_keypoints_with_one_bulk_transfer():
    boxes = _Boxes([[10.0, 20.0, 100.0, 200.0, 0.91, 0.0]])
    keypoints = _Keypoints(
        [[[11.5, 22.5, 0.8], [30.25, 40.75, 0.6]]]
    )

    records = model_manager._records_from_results([_Result(boxes, keypoints)])

    assert boxes.data.tolist_calls == 1
    assert keypoints.data.tolist_calls == 1
    assert records == [
        {
            "class_id": 0,
            "confidence": 0.91,
            "bbox": [10, 20, 100, 200],
            "keypoints": [[11.5, 22.5, 0.8], [30.25, 40.75, 0.6]],
        }
    ]


def test_result_records_handle_empty_results_without_transfer():
    assert model_manager._records_from_results([]) == []
    assert model_manager._records_from_results([_Result(None)]) == []

    boxes = _Boxes([])
    assert model_manager._records_from_results([_Result(boxes)]) == []
    assert boxes.data.tolist_calls == 1
