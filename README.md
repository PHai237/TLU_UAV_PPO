# TLU UAV PPO Simulation

Dự án mô phỏng bài toán điều hướng UAV trong khuôn viên Trường Đại học Thủy Lợi bằng bản đồ semantic 2D, môi trường Gymnasium và thuật toán PPO.

## Mục tiêu

- Xây dựng bản đồ mô phỏng có các khu vực chính: A1, Hội trường T45, Thư viện, K1, C1, B4, KTX số 4 và Cổng sau.
- Raster hóa bản đồ thành các lớp dữ liệu phục vụ thuật toán: occupancy, semantic type, height và risk.
- Định nghĩa môi trường MDP cho UAV: state, action, transition, reward và terminal condition.
- So sánh các phương pháp điều hướng: Random, Greedy, A* và PPO.
- Sinh ảnh/bảng phục vụ báo cáo và demo.

## Cấu trúc chính

```text
tlu_uav_ppo/
├── map_design/
│   └── generate_tlu_map.py          # Sinh semantic map và các raster layer
├── envs/
│   └── tlu_uav_env.py               # Gymnasium environment cho UAV
├── train/
│   └── train_ppo.py                 # Train PPO đa mục tiêu
├── eval/
│   ├── astar_map_demo.py            # A* baseline
│   ├── evaluate_policies.py         # Random/Greedy baseline
│   ├── evaluate_ppo.py              # PPO evaluation
│   ├── export_presentation.py       # Ảnh cho báo cáo
│   └── generate_report_summary.py   # Bảng/tóm tắt báo cáo
├── render/
│   └── live_demo.py                 # Demo trực quan
├── data/processed/                  # Dữ liệu map hiện hành
├── results/                         # Kết quả đánh giá và ảnh báo cáo
├── docs/                            # Tài liệu học/báo cáo
└── models/                          # Model PPO
```

## Chạy pipeline hiện hành

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

Sinh lại map:

```powershell
python map_design\generate_tlu_map.py
```

Chạy baseline và PPO evaluation:

```powershell
python eval\astar_map_demo.py
python eval\evaluate_policies.py
python eval\evaluate_ppo.py
```

Xuất ảnh và bảng cho báo cáo:

```powershell
python eval\export_presentation.py
python eval\generate_report_summary.py
```

Demo live:

```powershell
python render\live_demo.py --goal drop_t45 --policy ppo --hide-sensors
```

## Output nên dùng cho báo cáo

- `data/processed/previews/campus_render_v3.png`: bản đồ mô phỏng đẹp.
- `results/presentation/map_overview.png`: bản đồ tổng quan cho báo cáo.
- `results/presentation/raster_layers.png`: các lớp raster.
- `results/presentation/baseline_comparison.png`: so sánh baseline.
- `results/report_summary.md`: tóm tắt số liệu chính.
- `results/report_tables/*.csv`: bảng A*, policy comparison và best policy.

## Tài liệu học nhanh

- `docs/mdp_methodology.md`: giải thích MDP, state, action, reward, A*, PPO.
- `docs/run_and_study_guide.md`: flow chạy code và checklist ôn bảo vệ.
- `docs/output_manifest.md`: file nào là dữ liệu hiện hành, file nào đã xóa vì cũ.

## Ghi chú về xe động

Phiên bản hiện tại chưa triển khai xe động như vật cản cứng. Với giả định UAV bay khoảng 3m, xe mặt đất dưới 2m chỉ nên là ngữ cảnh hoặc risk thấp, không nên làm collision chính như tòa nhà/cây cao. Đây là hướng mở rộng phù hợp cho phần phát triển tiếp theo.
