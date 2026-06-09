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
│   ├── train_ppo.py                 # Train PPO đa mục tiêu
│   └── fine_tune_ppo.py             # Fine-tune model PPO đã có
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

Fine-tune PPO từ model hiện có nếu muốn thử cải thiện thêm:

```powershell
python train\fine_tune_ppo.py --timesteps 50000 --learning-rate 0.00005 --output-model models\ppo_tlu_uav_candidate
python eval\evaluate_ppo.py --model-path models\ppo_tlu_uav_candidate.zip --output-tag candidate --trajectory-prefix ppo_candidate
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
- `results/presentation/policy_comparison.png`: so sánh Random, Greedy và PPO.
- `results/report_summary.md`: tóm tắt số liệu chính.
- `results/report_tables/*.csv`: bảng A*, policy comparison và best policy.

## Tài liệu học nhanh

Đọc theo thứ tự này là dễ nắm dự án nhất:

- `docs/00_bat_dau_o_day.md`: mục lục học nhanh, dự án làm gì, cần hiểu gì.
- `docs/01_flow_chay_code_va_demo.md`: flow chạy code, lệnh demo, file nào sinh output nào.
- `docs/02_mdp_va_thuat_toan.md`: MDP, state, action, reward, A*, PPO.
- `docs/03_ket_qua_hien_tai.md`: kết quả hiện tại và cách giải thích PPO fail ở Thư viện.
- `docs/04_file_output_can_dung.md`: danh mục file/output dùng cho báo cáo.
- `docs/05_ghi_chu_ban_do.md`: ghi chú thiết kế bản đồ và giả định mô phỏng.

## Ghi chú về xe động

Phiên bản hiện tại chưa triển khai xe động như vật cản cứng. Với giả định UAV bay khoảng 3m, xe mặt đất dưới 2m chỉ nên là ngữ cảnh hoặc risk thấp, không nên làm collision chính như tòa nhà/cây cao. Đây là hướng mở rộng phù hợp cho phần phát triển tiếp theo.
