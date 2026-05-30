# TLU UAV PPO Map v1

Project này dùng Python để sinh bản đồ 2D semantic map cho đề tài UAV/drone PPO tại khu trường Thủy Lợi.

## Cấu trúc

```text
tlu_uav_ppo/
├── map_design/
│   └── generate_tlu_map.py        # Source code sinh map, sửa file này
├── data/
│   ├── raw/                       # Ảnh/tài liệu gốc nếu cần lưu
│   └── processed/                 # Output tự sinh, không sửa tay
│       ├── semantic_map.json
│       ├── pois.json
│       ├── dynamic_obstacles.json
│       ├── occupancy_grid.npy
│       ├── type_map.npy
│       ├── height_map.npy
│       ├── risk_map.npy
│       └── tlu_semantic_map_preview.png
├── envs/                          # Gymnasium env sau này
├── train/                         # PPO training scripts sau này
├── eval/                          # Benchmark/evaluation sau này
├── render/                        # Demo/video render sau này
├── models/
├── logs/
├── results/
└── docs/
```

## Chạy sinh map

```bash
cd tlu_uav_ppo
python -m pip install -r requirements.txt
python map_design/generate_tlu_map.py
```

## Layout hiện tại

- Canvas: 1400 x 900.
- Không lấy khu vòng tròn/cổng trước trước A1.
- A1 nằm bên trái.
- T45 là tòa lớn mái xanh nhạt ở phía trên/trung tâm.
- Thư viện nằm cạnh T45.
- K1 là cụm 2 tòa liền kề bên phải.
- C1 nằm giữa/dưới cụm Thư viện-K1.
- B4 Lab là khu no-fly có tường ngăn.
- Ký túc xá số 4 chỉ lấy đoạn đầu gần đường sinh viên đi.
- Cổng sau ở mép phải là pickup chính.
- Dropoff: T45, Thư viện, K1, C1, KTX số 4.

## Quy ước map

- `occupancy_grid.npy`: 1 là obstacle/no-fly, 0 là vùng có thể đi qua.
- `type_map.npy`: semantic type id.
- `height_map.npy`:
  - 0: free/flat
  - 1: hard obstacle
  - 2: medium/risky obstacle
  - 3: low/flyable obstacle
- `risk_map.npy`: risk từ 0 đến 1.
