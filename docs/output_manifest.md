# Manifest dữ liệu hiện hành

File này ghi rõ các output đang được dùng cho bản đồ và báo cáo hiện tại. Nếu cần dọn repo, ưu tiên giữ các file trong danh sách này.

## 1. Dữ liệu map hiện hành

Các file này được sinh bởi:

```powershell
python map_design\generate_tlu_map.py
```

| File | Ý nghĩa | Có đổi tên được không? |
| --- | --- | --- |
| `data/processed/semantic_map.json` | Mô tả semantic map dạng JSON: layer, vật thể, tòa nhà, cây, đường, sân | Không nên đổi nếu chưa sửa pipeline |
| `data/processed/pois.json` | Tọa độ pickup, dropoff, landmark | Không nên đổi |
| `data/processed/dynamic_obstacles.json` | Xe/obstacle động hoặc tĩnh ở phiên bản hiện tại | Không nên đổi |
| `data/processed/type_codes.json` | Bảng mã semantic type | Không nên đổi |
| `data/processed/occupancy_grid.npy` | Lưới vật cản dùng cho collision và A* | Không nên đổi |
| `data/processed/type_map.npy` | Lưới loại semantic dùng để render/eval | Không nên đổi |
| `data/processed/height_map.npy` | Lưới độ cao tương đối | Không nên đổi |
| `data/processed/risk_map.npy` | Lưới rủi ro dùng trong reward | Không nên đổi |

Lý do không đổi tên các file `.npy/.json` chính: `envs/tlu_uav_env.py`, `eval/astar_map_demo.py`, `eval/evaluate_policies.py` đang đọc đúng các đường dẫn này.

## 2. Ảnh preview map hiện hành

Các file này nằm trong `data/processed/previews/`:

| File | Dùng để làm gì |
| --- | --- |
| `campus_render_v3.png` | Ảnh bản đồ đẹp nhất để xem tổng quan |
| `occupancy_preview.png` | Minh họa vùng obstacle/free |
| `type_map_preview.png` | Minh họa semantic type |
| `height_map_preview.png` | Minh họa độ cao tương đối |
| `risk_map_preview.png` | Minh họa vùng risk |

## 3. Kết quả đánh giá hiện hành

Các file này được sinh bởi:

```powershell
python eval\astar_map_demo.py
python eval\evaluate_policies.py
python eval\evaluate_ppo.py
```

| File | Ý nghĩa |
| --- | --- |
| `results/astar_metrics.csv` | Bảng A* baseline |
| `results/astar_metrics.json` | Bảng A* dạng JSON |
| `results/policy_eval_summary.csv` | Random/Greedy baseline |
| `results/policy_eval_summary.json` | Random/Greedy baseline dạng JSON |
| `results/ppo_eval_summary_100k_radius30.csv` | PPO 100k đánh giá trên map hiện tại |
| `results/ppo_eval_summary_100k_radius30.json` | PPO 100k dạng JSON |
| `results/trajectories/*.png` | Ảnh quỹ đạo từng policy/goal |

## 4. Ảnh và bảng dùng cho báo cáo

Sinh bằng:

```powershell
python eval\export_presentation.py
python eval\generate_report_summary.py
```

| File | Ý nghĩa |
| --- | --- |
| `results/presentation/map_overview.png` | Bản đồ tổng quan cho slide/báo cáo |
| `results/presentation/raster_layers.png` | 4 lớp raster: occupancy/type/height/risk |
| `results/presentation/baseline_comparison.png` | So sánh baseline |
| `results/report_summary.md` | Tóm tắt số liệu chính để viết báo cáo |
| `results/report_tables/astar_routes.csv` | Bảng A* gọn |
| `results/report_tables/policy_comparison.csv` | Bảng so sánh policy gọn |
| `results/report_tables/goal_summary.csv` | Policy tốt nhất theo từng goal |

## 5. File cũ đã xóa

Các file dưới đây đã được xóa để tránh nhầm với map/kết quả hiện tại:

| File | Lý do xóa |
| --- | --- |
| `data/processed/tlu_semantic_map_preview.png` | Tên preview cũ, không còn đúng pipeline hiện tại |
| `results/ppo_eval_summary_50k_radius30.csv` | Kết quả PPO 50k cũ, không phải evaluation chính trên map cuối |
| `results/ppo_eval_summary_50k_radius30.json` | Kết quả PPO 50k cũ |
| `models/ppo_tlu_uav_50k_radius30_evalonly.zip` | Model eval-only cũ, dễ gây nhầm với PPO 100k hiện hành |

## 6. File docs đang để riêng

Các file sau đang là ghi chú riêng, không được dùng làm output chính của pipeline:

| File | Ghi chú |
| --- | --- |
| `docs/map_notes.md` | Ghi chú map, hiện có thay đổi riêng |
| `docs/current_results.md` | Ghi chú kết quả cũ/riêng, không phải report summary hiện hành |
