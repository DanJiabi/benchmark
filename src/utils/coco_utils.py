"""COCO 数据集相关工具函数."""


def yolo_index_to_coco_id(yolo_index: int) -> int:
    """
    将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）

    COCO 数据集的类别 ID 不是连续的 1-80，而是有一些跳跃的 1-90。
    例如：YOLO 的索引 11（stop sign）对应 COCO 的 ID 13（跳过了 12）。

    Args:
        yolo_index: YOLO 类别索引（0-79）

    Returns:
        COCO 类别 ID（1-90）

    Example:
        >>> yolo_index_to_coco_id(0)  # person
        1
        >>> yolo_index_to_coco_id(11)  # stop sign
        13
        >>> yolo_index_to_coco_id(79)  # toothbrush
        90
    """
    coco_id_mapping = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    if 0 <= yolo_index < len(coco_id_mapping):
        return coco_id_mapping[yolo_index]
    return yolo_index + 1  # 默认值（向后兼容）


def get_coco_category_names() -> dict:
    """
    获取 COCO 类别名称（按 YOLO 索引 0-79）

    Returns:
        字典，键为 YOLO 索引（0-79），值为类别名称
    """
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }
