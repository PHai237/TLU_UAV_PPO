from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath


# ============================================================
# TLU UAV PPO - Semantic Campus Map V3
# Scope: A1 -> T45/Library -> K1/C1 -> Dormitory 4 -> Back Gate
# Canvas: 1400 x 900
# Output: semantic_map.json, pois.json, dynamic_obstacles.json,
#         occupancy_grid.npy, type_map.npy, height_map.npy, risk_map.npy,
#         and preview PNG files.
# ============================================================

WIDTH = 1400
HEIGHT = 900

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREVIEW_DIR = PROCESSED_DIR / "previews"

TYPE_CODES = {
    "road": 0,
    "building": 1,
    "green": 2,
    "yard": 3,
    "no_fly": 4,
    "pickup": 5,
    "dropoff": 6,
}

TYPE_COLORS = {
    "road": "#b8bdc2",
    "road_edge": "#7f8c8d",
    "building": "#b56a5a",
    "building_dark": "#7a3f35",
    "roof_green": "#b7d6c0",
    "roof_gray": "#b0a7a2",
    "green": "#3f8f4f",
    "green_dark": "#276738",
    "yard": "#edf1df",
    "paving": "#e8cfc1",
    "no_fly": "#e25d5d",
    "wall": "#9e4c4c",
    "pickup": "#f4b942",
    "dropoff": "#2878bd",
    "vehicle": "#4b4b4b",
    "background": "#dfe8d2",
}

HEIGHT_COLORS = [
    "#ffffff",  # 0 normal
    "#202020",  # 1 high / hard obstacle
    "#f39c12",  # 2 medium / risky
    "#90caf9",  # 3 low / flyable
]

TYPE_PREVIEW_COLORS = [
    "#b8bdc2",  # road
    "#b56a5a",  # building
    "#3f8f4f",  # green
    "#edf1df",  # yard
    "#e25d5d",  # no_fly
    "#f4b942",  # pickup
    "#2878bd",  # dropoff
]


# ============================================================
# Geometry helpers
# ============================================================

def rect(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def poly(points: list[list[float]]) -> list[list[float]]:
    return points


def oval(cx: float, cy: float, rx: float, ry: float, n: int = 24) -> list[list[float]]:
    return [
        [
            cx + rx * np.cos(2.0 * np.pi * i / n),
            cy + ry * np.sin(2.0 * np.pi * i / n),
        ]
        for i in range(n)
    ]


def item(
    id_: str,
    label: str,
    cls: str,
    points: list[list[float]],
    *,
    height_level: int = 0,
    estimated_height_m: float | None = None,
    subclass: str | None = None,
    risk_weight: float = 0.0,
    render_color: str | None = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "id": id_,
        "label": label,
        "class": cls,
        "points": points,
        "height_level": int(height_level),
        "risk_weight": float(risk_weight),
    }
    if estimated_height_m is not None:
        obj["estimated_height_m"] = float(estimated_height_m)
    if subclass is not None:
        obj["subclass"] = subclass
    if render_color is not None:
        obj["render_color"] = render_color
    return obj


def dorm4_front_tree_items() -> list[dict[str, Any]]:
    """Small tree crowns around the dormitory 4 front yard, leaving a door corridor."""
    centers = [
        (710, 525, 22), (770, 525, 21), (830, 525, 22), (890, 525, 21),
        (710, 585, 22), (890, 585, 22),
        (710, 645, 22), (890, 645, 22),
        (1105, 525, 21), (1165, 525, 21), (1225, 525, 21), (1285, 525, 21),
        (1105, 585, 21), (1285, 585, 21),
        (1105, 645, 21), (1285, 645, 21),
    ]

    trees = []
    for index, (x, y, radius) in enumerate(centers, start=1):
        trees.append(
            item(
                f"tree_medium_dorm4_front_{index:02d}",
                "Cây khu bãi đất trước KTX số 4",
                "green",
                oval(x, y, radius, radius, n=20),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.50,
            )
        )
    return trees


def polygon_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)

    min_x = max(int(np.floor(pts[:, 0].min())), 0)
    max_x = min(int(np.ceil(pts[:, 0].max())), width - 1)
    min_y = max(int(np.floor(pts[:, 1].min())), 0)
    max_y = min(int(np.ceil(pts[:, 1].max())), height - 1)

    xs = np.arange(min_x, max_x + 1) + 0.5
    ys = np.arange(min_y, max_y + 1) + 0.5
    gx, gy = np.meshgrid(xs, ys)

    sample_points = np.column_stack([gx.ravel(), gy.ravel()])
    path = MplPath(pts)
    small_mask = path.contains_points(sample_points).reshape(len(ys), len(xs))

    mask = np.zeros((height, width), dtype=bool)
    mask[min_y:max_y + 1, min_x:max_x + 1] = small_mask
    return mask


def disk_mask(x: float, y: float, radius: float, width: int, height: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2


# ============================================================
# Semantic map V3
# Layout after latest corrections:
# - Do NOT include the front-gate circular courtyard before A1.
# - A1 is on the left.
# - T45 is a large horizontal light-green-roof building upper/central.
# - Library is a low light-green-roof block attached/next to T45.
# - K1 is two adjacent blocks on the right.
# - C1 is lower/central, forming a triangle with Library and K1.
# - B4 lab is lower-left, separated by wall/no-fly.
# - Dormitory 4 is only the first red segment near student path.
# - Back gate is the right-edge intersection and main pickup.
# ============================================================

layers: list[dict[str, Any]] = [
    {
        "name": "roads",
        "items": [
            # Main inner road from A1 side to the back-gate area.
            item(
                "road_main_horizontal",
                "Main Internal Road",
                "road",
                rect(245, 455, 1400, 505),
            ),
            # Road in front of T45 / library.
            item(
                "road_t45_front",
                "Road in front of T45 - Library",
                "road",
                rect(315, 250, 1040, 305),
            ),
            # Left vertical road along A1, but not including the front circular courtyard.
            item(
                "road_a1_side",
                "A1 Side Road",
                "road",
                rect(245, 0, 315, 735),
            ),
            # Road around the main courtyard.
            item(
                "road_courtyard_bottom",
                "Courtyard Bottom Road",
                "road",
                rect(315, 660, 690, 705),
            ),
            item(
                "road_courtyard_right",
                "Courtyard Right Road",
                "road",
                rect(635, 305, 690, 660),
            ),
            item(
                "road_courtyard_left",
                "Courtyard Left Road",
                "road",
                rect(315, 305, 365, 660),
            ),
            # Road between C1 and K1. Keep this vertical like the reference plan;
            # do not draw an artificial diagonal/triangle road through C1.
            item(
                "road_c1_k1_vertical",
                "C1 - K1 Vertical Road",
                "road",
                rect(960, 250, 1015, 505),
            ),
            # Road to back gate on the right edge.
            item(
                "road_back_gate",
                "Back Gate Road",
                "road",
                rect(1325, 60, 1370, 500),
            ),
            # Road along B4 separator wall.
            item(
                "road_b4_boundary",
                "Road near B4 Wall",
                "road",
                rect(245, 700, 560, 720),
            ),
        ],
    },
    {
        "name": "yards",
        "items": [
            item(
                "yard_main_courtyard",
                "Sân trước T45",
                "yard",
                rect(365, 320, 635, 445),
                risk_weight=0.08,
                render_color=TYPE_COLORS["paving"],
            ),
            item(
                "yard_triangle_open_space",
                "Open Space: Library - K1 - C1",
                "yard",
                rect(690, 305, 960, 430),
                risk_weight=0.10,
                render_color="#ead9c6",
            ),
            item(
                "yard_dorm4_front",
                "Bãi đất trước KTX số 4",
                "yard",
                rect(690, 505, 1305, 735),
                risk_weight=0.18,
                render_color="#ead9c6",
            ),
            item(
                "yard_west_inside",
                "A1 Inner Yard",
                "yard",
                rect(315, 130, 365, 250),
                risk_weight=0.10,
            ),
        ],
    },
    {
        "name": "buildings",
        "items": [
            item(
                "building_a1",
                "A1",
                "building",
                rect(75, 0, 240, 685),
                height_level=1,
                estimated_height_m=16,
                subclass="academic_building",
                risk_weight=1.0,
                render_color="#c87868",
            ),
            item(
                "building_t45",
                "Hội trường T45",
                "building",
                rect(365, 50, 635, 220),
                height_level=1,
                estimated_height_m=14,
                subclass="hall_light_green_roof",
                risk_weight=1.0,
                render_color=TYPE_COLORS["roof_green"],
            ),
            item(
                "building_library",
                "Thư viện",
                "building",
                rect(635, 50, 960, 235),
                height_level=1,
                estimated_height_m=14,
                subclass="library_low_light_green_roof",
                risk_weight=1.0,
                render_color="#c4dfcf",
            ),
            # K1 is a two-block cluster on the right.
            item(
                "building_k1_north",
                "K1",
                "building",
                rect(1085, 80, 1325, 220),
                height_level=1,
                estimated_height_m=30,
                subclass="k1_cluster",
                risk_weight=1.0,
                render_color="#d7c8ba",
            ),
            item(
                "building_k1_tower",
                "",
                "building",
                rect(1130, 220, 1210, 315),
                height_level=1,
                estimated_height_m=35,
                subclass="k1_tower",
                risk_weight=1.0,
                render_color="#d7c8ba",
            ),
            item(
                "building_k1_south",
                "",
                "building",
                rect(1085, 315, 1325, 430),
                height_level=1,
                estimated_height_m=30,
                subclass="k1_cluster",
                risk_weight=1.0,
                render_color="#d7c8ba",
            ),
            item(
                "building_c1",
                "C1",
                "building",
                rect(735, 345, 880, 445),
                height_level=1,
                estimated_height_m=12,
                subclass="academic_building",
                risk_weight=1.0,
                render_color="#c87868",
            ),
            item(
                "building_c1_wing",
                "C1 Wing",
                "building",
                rect(700, 375, 770, 430),
                height_level=1,
                estimated_height_m=12,
                subclass="academic_building",
                risk_weight=1.0,
                render_color="#c87868",
            ),
            item(
                "building_b4_lab",
                "B4",
                "building",
                rect(360, 735, 500, 790),
                height_level=1,
                estimated_height_m=10,
                subclass="lab_inactive",
                risk_weight=1.0,
                render_color="#9d7a6b",
            ),
            item(
                "building_dorm4_first",
                "KTX số 4",
                "building",
                rect(700, 740, 1225, 825),
                height_level=1,
                estimated_height_m=18,
                subclass="dormitory_first_segment",
                risk_weight=1.0,
                render_color="#b94f45",
            ),
        ],
    },
    {
        "name": "walls_no_fly",
        "items": [
            item(
                "wall_b4_separator",
                "Tường ngăn B4",
                "no_fly",
                rect(245, 755, 275, 900),
                height_level=1,
                estimated_height_m=4,
                subclass="separator_wall",
                risk_weight=1.0,
                render_color=TYPE_COLORS["wall"],
            ),
            item(
                "no_fly_inside_b4",
                "No-fly: B4 Area",
                "no_fly",
                rect(40, 760, 245, 900),
                height_level=1,
                estimated_height_m=10,
                subclass="inactive_lab_area",
                risk_weight=1.0,
                render_color="#d96b6b",
            ),
        ],
    },
    {
        "name": "trees_high",
        "items": [
            item(
                "tree_high_a1_west",
                "Cây cao cạnh A1",
                "green",
                oval(35, 140, 45, 60),
                height_level=1,
                estimated_height_m=8,
                subclass="tree_high",
                risk_weight=0.95,
            ),
            item(
                "tree_high_between_a1_t45",
                "Cây cao giữa A1 - T45",
                "green",
                oval(330, 245, 45, 45),
                height_level=1,
                estimated_height_m=8,
                subclass="tree_high",
                risk_weight=0.90,
            ),
            item(
                "tree_high_k1_dorm_side",
                "Cây cao cạnh K1/KTX",
                "green",
                oval(1290, 590, 24, 24),
                height_level=1,
                estimated_height_m=8,
                subclass="tree_high",
                risk_weight=0.90,
            ),
            item(
                "tree_high_b4_front",
                "Cây cao trước B4",
                "green",
                oval(420, 830, 55, 38),
                height_level=1,
                estimated_height_m=8,
                subclass="tree_high",
                risk_weight=0.90,
            ),
        ],
    },
    {
        "name": "trees_medium",
        "items": [
            item(
                "tree_medium_front_t45_left",
                "Cây trung bình trước T45 - trái",
                "green",
                oval(435, 350, 1, 1),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.0,
            ),
            item(
                "tree_medium_front_t45_right",
                "Cây trung bình trước T45 - phải",
                "green",
                oval(905, 305, 55, 45),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.55,
            ),
            item(
                "tree_medium_library_c1",
                "Cây giữa Thư viện và C1",
                "green",
                oval(780, 305, 75, 50),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.60,
            ),
            item(
                "tree_medium_k1_c1",
                "Cây giữa K1 và C1",
                "green",
                oval(930, 510, 24, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.60,
            ),
            item(
                "tree_medium_lower_garden_north",
                "Hàng cây mép trên ô đất dưới vườn hoa",
                "green",
                oval(510, 525, 95, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.45,
            ),
            item(
                "tree_medium_lower_garden_west",
                "Hàng cây mép trái ô đất dưới vườn hoa",
                "green",
                oval(375, 585, 28, 65),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.45,
            ),
            item(
                "tree_medium_lower_garden_south",
                "Hàng cây mép dưới ô đất dưới vườn hoa",
                "green",
                oval(505, 655, 115, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.45,
            ),
            item(
                "tree_medium_lower_c1_north",
                "Hàng cây mép trên ô đất dưới C1",
                "green",
                oval(725, 525, 24, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.50,
            ),
            item(
                "tree_medium_lower_c1_west",
                "Hàng cây mép trái ô đất dưới C1",
                "green",
                oval(725, 585, 24, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.50,
            ),
            item(
                "tree_medium_lower_c1_south",
                "Hàng cây mép dưới ô đất dưới C1",
                "green",
                oval(725, 645, 24, 24),
                height_level=2,
                estimated_height_m=3,
                subclass="tree_medium",
                risk_weight=0.50,
            ),
            *dorm4_front_tree_items(),
        ],
    },
    {
        "name": "trees_low_gardens",
        "items": [
            item(
                "garden_low_1",
                "Bồn cây thấp 1",
                "green",
                rect(390, 340, 455, 375),
                height_level=3,
                estimated_height_m=1.2,
                subclass="tree_low",
                risk_weight=0.12,
                render_color="#77b255",
            ),
            item(
                "garden_low_2",
                "Bồn cây thấp 2",
                "green",
                rect(545, 340, 610, 375),
                height_level=3,
                estimated_height_m=1.2,
                subclass="tree_low",
                risk_weight=0.12,
                render_color="#77b255",
            ),
            item(
                "garden_low_3",
                "Bồn cây thấp 3",
                "green",
                rect(390, 400, 455, 435),
                height_level=3,
                estimated_height_m=1.2,
                subclass="tree_low",
                risk_weight=0.12,
                render_color="#77b255",
            ),
            item(
                "garden_low_4",
                "Bồn cây thấp 4",
                "green",
                rect(545, 400, 610, 435),
                height_level=3,
                estimated_height_m=1.2,
                subclass="tree_low",
                risk_weight=0.12,
                render_color="#77b255",
            ),
            item(
                "garden_low_center",
                "Bồn cây thấp giữa sân",
                "green",
                oval(505, 388, 42, 22),
                height_level=3,
                estimated_height_m=1.2,
                subclass="tree_low",
                risk_weight=0.12,
                render_color="#77b255",
            ),
        ],
    },
]

pois = {
    "pickup_points": [
        {
            "id": "pickup_back_gate",
            "label": "Pickup - Cổng sau",
            "x": 1350,
            "y": 490,
        }
    ],
    "dropoff_points": [
        {"id": "drop_t45", "label": "Drop - T45", "x": 590, "y": 245},
        {"id": "drop_library", "label": "Drop - Thư viện", "x": 870, "y": 275},
        {"id": "drop_k1", "label": "Drop - K1", "x": 1055, "y": 365},
        {"id": "drop_c1", "label": "Drop - C1", "x": 675, "y": 450},
        {"id": "drop_dorm4", "label": "Drop - KTX 4", "x": 1030, "y": 720},
    ],
    "landmarks": [
        {"id": "lm_a1", "label": "A1", "x": 170, "y": 420},
        {"id": "lm_a1_door", "label": "Cửa A1", "x": 245, "y": 278},
        {"id": "lm_t45", "label": "T45", "x": 590, "y": 185},
        {"id": "lm_library", "label": "Thư viện", "x": 870, "y": 185},
        {"id": "lm_k1", "label": "K1", "x": 1205, "y": 300},
        {"id": "lm_c1", "label": "C1", "x": 910, "y": 550},
        {"id": "lm_b4", "label": "B4", "x": 170, "y": 810},
        {"id": "lm_dorm4", "label": "KTX số 4", "x": 1180, "y": 790},
        {"id": "lm_back_gate", "label": "Cổng sau", "x": 1350, "y": 490},
    ],
}

POI_OVERRIDES = {
    "pickup_back_gate": ("Cổng sau", 1350, 490),
    "drop_t45": ("Cửa T45", 500, 245),
    "drop_library": ("Cửa thư viện", 985, 145),
    "drop_k1": ("Cửa K1", 1105, 268),
    "drop_c1": ("Cửa C1", 715, 446),
    "drop_dorm4": ("Cửa KTX số 4", 962, 722),
    "lm_a1": ("A1", 170, 420),
    "lm_a1_door": ("Cửa A1", 260, 278),
    "lm_t45": ("Hội trường T45", 500, 135),
    "lm_library": ("Thư viện", 800, 135),
    "lm_k1": ("K1", 1205, 205),
    "lm_c1": ("C1", 820, 400),
    "lm_b4": ("B4", 430, 762),
    "lm_dorm4": ("KTX số 4", 960, 780),
    "lm_back_gate": ("Cổng sau", 1350, 490),
}

for group_name in ("pickup_points", "dropoff_points", "landmarks"):
    for point in pois[group_name]:
        override = POI_OVERRIDES.get(point["id"])
        if override is None:
            continue

        point["label"], point["x"], point["y"] = override

# Vehicles are static in V3. Later they can become dynamic obstacles.
dynamic_obstacles = {
    "vehicles": [
        # Cars near A1 side / left internal road.
        {"id": "car_a1_01", "type": "car", "x": 315, "y": 205, "w": 16, "h": 30, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_a1_02", "type": "car", "x": 315, "y": 270, "w": 16, "h": 30, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_a1_03", "type": "car", "x": 315, "y": 610, "w": 16, "h": 30, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        # Cars in front road and near T45.
        {"id": "car_front_01", "type": "car", "x": 485, "y": 700, "w": 32, "h": 16, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_front_02", "type": "car", "x": 565, "y": 700, "w": 32, "h": 16, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_front_03", "type": "car", "x": 645, "y": 700, "w": 32, "h": 16, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_t45_01", "type": "car", "x": 835, "y": 500, "w": 16, "h": 32, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        # Vehicles near back gate / K1 / C1 area.
        {"id": "car_gate_01", "type": "car", "x": 1160, "y": 490, "w": 32, "h": 16, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
        {"id": "car_gate_02", "type": "car", "x": 1240, "y": 490, "w": 32, "h": 16, "angle": 0, "height_m": 1.6, "height_level": 3, "dynamic": False},
    ]
}

semantic_map = {
    "map_name": "TLU UAV Campus Map V3 - A1 T45 Library K1 C1 B4 Dorm4 Back Gate",
    "width": WIDTH,
    "height": HEIGHT,
    "background": "yard",
    "note": (
        "Approximate hand-designed 2D semantic map based on user-provided top-down campus screenshots. "
        "Coordinates are simulation coordinates, not GPS/survey coordinates. "
        "The front-gate circular courtyard before A1 is intentionally excluded."
    ),
    "layers": layers,
}


# ============================================================
# Build raster maps
# ============================================================

def vehicle_mask(vehicle: dict[str, Any]) -> np.ndarray:
    # For now use an axis-aligned rectangle around the vehicle.
    x = float(vehicle["x"])
    y = float(vehicle["y"])
    w = float(vehicle["w"])
    h = float(vehicle["h"])
    return polygon_mask(rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2), WIDTH, HEIGHT)


def build_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    type_map = np.full((HEIGHT, WIDTH), TYPE_CODES["yard"], dtype=np.uint8)
    occupancy_grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    height_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    risk_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    # Draw order follows layer order. Later layers can overwrite type/height/risk.
    for layer in semantic_map["layers"]:
        for obj in layer["items"]:
            if obj["id"] in {"wall_b4_separator", "no_fly_inside_b4", "tree_medium_front_t45_left"}:
                continue

            mask = polygon_mask(obj["points"], WIDTH, HEIGHT)
            cls = obj["class"]
            h_level = int(obj.get("height_level", 0))
            risk = float(obj.get("risk_weight", 0.0))

            type_map[mask] = TYPE_CODES[cls]
            height_map[mask] = h_level
            risk_map[mask] = np.maximum(risk_map[mask], risk)

            if cls in {"building", "no_fly"} or h_level == 1:
                occupancy_grid[mask] = 1

    # IMPORTANT: second pass for roads.
    # Roads should remain traversable even if a green/tree zone was drawn later.
    # But roads must NOT cut through buildings or no-fly walls.
    hard_structure_mask = (type_map == TYPE_CODES["building"]) | (type_map == TYPE_CODES["no_fly"])

    for layer in semantic_map["layers"]:
        for obj in layer["items"]:
            if obj["id"] in {"wall_b4_separator", "no_fly_inside_b4", "tree_medium_front_t45_left"}:
                continue

            if obj["class"] != "road":
                continue

            road_mask = polygon_mask(obj["points"], WIDTH, HEIGHT)
            clear_mask = road_mask & (~hard_structure_mask)

            type_map[clear_mask] = TYPE_CODES["road"]
            occupancy_grid[clear_mask] = 0
            height_map[clear_mask] = 0
            risk_map[clear_mask] = 0.0

    # Vehicles are not hard-blocked in static map V3, but they affect height/risk.
    for vehicle in dynamic_obstacles["vehicles"]:
        mask = vehicle_mask(vehicle)
        h_level = int(vehicle["height_level"])
        height_map[mask] = np.maximum(height_map[mask], h_level)

        if h_level == 3:
            risk_map[mask] = np.maximum(risk_map[mask], 0.25)
        elif h_level == 2:
            risk_map[mask] = np.maximum(risk_map[mask], 0.65)
        else:
            risk_map[mask] = np.maximum(risk_map[mask], 1.0)

    # POIs are always free and visible.
    for p in pois["pickup_points"]:
        mask = disk_mask(float(p["x"]), float(p["y"]), 10, WIDTH, HEIGHT)
        type_map[mask] = TYPE_CODES["pickup"]
        occupancy_grid[mask] = 0
        height_map[mask] = 0
        risk_map[mask] = 0.0

    for p in pois["dropoff_points"]:
        mask = disk_mask(float(p["x"]), float(p["y"]), 10, WIDTH, HEIGHT)
        type_map[mask] = TYPE_CODES["dropoff"]
        occupancy_grid[mask] = 0
        height_map[mask] = 0
        risk_map[mask] = 0.0

    return occupancy_grid, type_map, height_map, risk_map


# ============================================================
# Pretty rendering
# ============================================================

def add_shadowed_polygon(
    ax,
    points,
    facecolor,
    edgecolor="#333333",
    linewidth=1.0,
    alpha=1.0,
    zorder=1,
    shadow_alpha=0.10,
):
    pts = np.asarray(points, dtype=float)
    if shadow_alpha > 0.0:
        shadow_pts = pts + np.array([4.0, 4.0])
        shadow = Polygon(
            shadow_pts,
            closed=True,
            facecolor="black",
            edgecolor="none",
            alpha=shadow_alpha,
            zorder=zorder - 0.1,
        )
        ax.add_patch(shadow)
    patch = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


def add_label(ax, x, y, text, *, size=8, color="#222222", weight="bold", zorder=20):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        weight=weight,
        zorder=zorder,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )


def draw_vehicle(ax, vehicle: dict[str, Any]):
    x = float(vehicle["x"])
    y = float(vehicle["y"])
    w = float(vehicle["w"])
    h = float(vehicle["h"])
    h_level = int(vehicle["height_level"])
    color = "#5d5d5d" if h_level == 3 else "#303030"

    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=4",
        facecolor=color,
        edgecolor="#202020",
        linewidth=0.8,
        alpha=0.95,
        zorder=12,
    )
    ax.add_patch(patch)


def draw_poi(ax, p: dict[str, Any], kind: str):
    x = float(p["x"])
    y = float(p["y"])
    if kind == "pickup":
        ax.scatter(x, y, s=160, marker="o", color=TYPE_COLORS["pickup"], edgecolors="black", linewidths=1.2, zorder=30)
    else:
        ax.scatter(x, y, s=170, marker="X", color=TYPE_COLORS["dropoff"], edgecolors="black", linewidths=1.2, zorder=30)
    ax.text(
        x + 14,
        y - 12,
        p["label"],
        fontsize=8,
        weight="bold",
        zorder=31,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def draw_door_marker(ax, p: dict[str, Any]):
    x = float(p["x"])
    y = float(p["y"])
    ax.scatter(x, y, s=145, marker="X", color=TYPE_COLORS["dropoff"], edgecolors="black", linewidths=1.1, zorder=29)
    ax.text(
        x + 14,
        y - 10,
        p["label"],
        fontsize=8,
        weight="bold",
        zorder=30,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def save_pretty_preview() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)
    ax.set_facecolor(TYPE_COLORS["background"])

    zorders = {
        "roads": 14,
        "yards": 3,
        "buildings": 8,
        "walls_no_fly": 10,
        "trees_high": 5,
        "trees_medium": 5,
        "trees_low_gardens": 6,
    }

    # Draw roads first with edges.
    for layer in semantic_map["layers"]:
        layer_name = layer["name"]
        z = zorders.get(layer_name, 5)
        for obj in layer["items"]:
            if obj["id"] in {"wall_b4_separator", "no_fly_inside_b4", "tree_medium_front_t45_left"}:
                continue

            cls = obj["class"]
            color = obj.get("render_color")
            if color is None:
                if cls == "road":
                    color = TYPE_COLORS["road"]
                elif cls == "yard":
                    color = TYPE_COLORS["yard"]
                elif cls == "building":
                    color = TYPE_COLORS["building"]
                elif cls == "green":
                    color = TYPE_COLORS["green"]
                elif cls == "no_fly":
                    color = TYPE_COLORS["no_fly"]
                else:
                    color = "#cccccc"

            if cls == "road":
                add_shadowed_polygon(
                    ax,
                    obj["points"],
                    color,
                    edgecolor=color,
                    linewidth=0.2,
                    alpha=1.0,
                    zorder=z,
                    shadow_alpha=0.0,
                )
            elif cls == "yard":
                add_shadowed_polygon(ax, obj["points"], color, edgecolor="#b9b09b", linewidth=0.9, alpha=0.98, zorder=z)
            elif cls == "building":
                add_shadowed_polygon(ax, obj["points"], color, edgecolor=TYPE_COLORS["building_dark"], linewidth=1.2, alpha=1.0, zorder=z)
            elif cls == "no_fly":
                add_shadowed_polygon(ax, obj["points"], color, edgecolor="#8e2f2f", linewidth=1.0, alpha=0.72, zorder=z)
            elif cls == "green":
                # High/medium/low greens get different opacity and edge treatment.
                h_level = int(obj.get("height_level", 0))
                alpha = 0.95 if h_level == 1 else 0.82 if h_level == 2 else 0.72
                edge = TYPE_COLORS["green_dark"] if h_level in {1, 2} else "#5f9843"
                add_shadowed_polygon(ax, obj["points"], color, edgecolor=edge, linewidth=0.8, alpha=alpha, zorder=z)

            # Label important objects only.
            if cls in {"building", "no_fly"} and obj["label"] and not obj["id"].endswith("_wing"):
                pts = np.asarray(obj["points"], dtype=float)
                add_label(ax, float(pts[:, 0].mean()), float(pts[:, 1].mean()), obj["label"], size=8)

    # Decorative garden paths, matching the central garden in the reference plan.
    ax.plot([390, 610], [388, 388], color="#ef9a9a", linewidth=1.4, linestyle="--", zorder=15)
    ax.plot([505, 505], [338, 438], color="#ef9a9a", linewidth=1.4, linestyle="--", zorder=15)
    courtyard_center = Circle((505, 388), 22, facecolor="#9dcc76", edgecolor="#5f9843", linewidth=0.8, alpha=0.95, zorder=16)
    ax.add_patch(courtyard_center)

    # Vehicles.
    for vehicle in dynamic_obstacles["vehicles"]:
        draw_vehicle(ax, vehicle)

    # POIs.
    for p in pois["pickup_points"]:
        draw_poi(ax, p, "pickup")
    for p in pois["dropoff_points"]:
        draw_poi(ax, p, "dropoff")
    for p in pois["landmarks"]:
        if p["id"].endswith("_door"):
            draw_door_marker(ax, p)

    # Legend as small custom patches.
    legend_items = [
        ("Đường đi", TYPE_COLORS["road"]),
        ("Sân / nền", TYPE_COLORS["paving"]),
        ("Tòa nhà", TYPE_COLORS["building"]),
        ("Cây xanh", TYPE_COLORS["green"]),
        ("Cổng sau", TYPE_COLORS["pickup"]),
        ("Điểm giao", TYPE_COLORS["dropoff"]),
    ]
    lx, ly = 25, HEIGHT - len(legend_items) * 26 - 35
    for i, (name, color) in enumerate(legend_items):
        y = ly + i * 26
        ax.add_patch(Rectangle((lx, y), 22, 14, facecolor=color, edgecolor="black", linewidth=0.5, zorder=50))
        ax.text(lx + 30, y + 7, name, va="center", fontsize=8, zorder=50)

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_aspect("equal")
    ax.set_title("Bản đồ mô phỏng khuôn viên Trường Đại học Thủy Lợi", fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8, colors="#666666")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "campus_render_v3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Save outputs
# ============================================================

def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_map_previews(occupancy_grid: np.ndarray, type_map: np.ndarray, height_map: np.ndarray, risk_map: np.ndarray) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    ax.imshow(occupancy_grid, cmap="gray_r", origin="upper", interpolation="nearest")
    ax.set_title("Occupancy Grid: white = free, black = blocked")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "occupancy_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    cmap_type = ListedColormap(TYPE_PREVIEW_COLORS)
    ax.imshow(type_map, cmap=cmap_type, origin="upper", vmin=0, vmax=len(TYPE_PREVIEW_COLORS) - 1, interpolation="nearest")
    ax.set_title("Type Map Preview")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "type_map_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    cmap_height = ListedColormap(HEIGHT_COLORS)
    ax.imshow(height_map, cmap=cmap_height, origin="upper", vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("Height Map: 0 normal, 1 hard, 2 risky, 3 low")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "height_map_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    im = ax.imshow(risk_map, origin="upper", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Risk Map Preview")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "risk_map_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    occupancy_grid, type_map, height_map, risk_map = build_maps()

    save_json(PROCESSED_DIR / "semantic_map.json", semantic_map)
    save_json(PROCESSED_DIR / "pois.json", pois)
    save_json(PROCESSED_DIR / "dynamic_obstacles.json", dynamic_obstacles)

    type_codes = {
        "type_codes": TYPE_CODES,
        "height_level_meaning": {
            "0": "normal / clear area",
            "1": "high obstacle, cannot fly over at around 3m",
            "2": "medium/risky obstacle, avoid if possible",
            "3": "low obstacle, flyable at around 3m",
        },
        "risk_map_meaning": "0.0 = safe, 1.0 = highly risky",
        "map_size": {"width": WIDTH, "height": HEIGHT},
        "main_pickup": "pickup_back_gate",
        "dropoffs": [p["id"] for p in pois["dropoff_points"]],
    }
    save_json(PROCESSED_DIR / "type_codes.json", type_codes)

    np.save(PROCESSED_DIR / "occupancy_grid.npy", occupancy_grid)
    np.save(PROCESSED_DIR / "type_map.npy", type_map)
    np.save(PROCESSED_DIR / "height_map.npy", height_map)
    np.save(PROCESSED_DIR / "risk_map.npy", risk_map)

    save_pretty_preview()
    save_map_previews(occupancy_grid, type_map, height_map, risk_map)

    print("Generated TLU UAV map V3 successfully.")
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Processed outputs : {PROCESSED_DIR}")
    print(f"Preview images    : {PREVIEW_DIR}")
    print("Files generated:")
    print("- semantic_map.json")
    print("- pois.json")
    print("- dynamic_obstacles.json")
    print("- type_codes.json")
    print("- occupancy_grid.npy")
    print("- type_map.npy")
    print("- height_map.npy")
    print("- risk_map.npy")
    print("- previews/campus_render_v3.png")
    print("- previews/occupancy_preview.png")
    print("- previews/type_map_preview.png")
    print("- previews/height_map_preview.png")
    print("- previews/risk_map_preview.png")


if __name__ == "__main__":
    main()
