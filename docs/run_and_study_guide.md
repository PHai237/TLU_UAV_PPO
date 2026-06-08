# Flow chạy code và checklist ôn bảo vệ

File này là lộ trình học nhanh để nắm code và trình bày dự án. Nếu chỉ còn ít ngày, hãy học theo thứ tự dưới đây.

## 1. Flow tổng quát của dự án

```text
generate_tlu_map.py
    -> semantic_map.json / pois.json
    -> occupancy_grid.npy / type_map.npy / height_map.npy / risk_map.npy
    -> preview images

tlu_uav_env.py
    -> đọc các file map đã sinh
    -> tạo Gymnasium environment
    -> định nghĩa state, action, reward, done

train_ppo.py
    -> dùng environment để train PPO
    -> lưu model vào models/

evaluate_policies.py
    -> chạy Random và Greedy baseline
    -> lưu bảng kết quả + ảnh trajectory

astar_map_demo.py
    -> chạy A* trên occupancy grid
    -> lưu metrics + ảnh đường đi

evaluate_ppo.py
    -> load model PPO
    -> đánh giá PPO trên 5 goal

export_presentation.py
    -> sinh ảnh đẹp cho báo cáo

generate_report_summary.py
    -> gom bảng kết quả thành report_summary.md
```

## 2. Lệnh chạy từ đầu đến cuối

Chạy từ thư mục gốc project:

```powershell
python -m pip install -r requirements.txt
```

Sinh lại bản đồ:

```powershell
python map_design\generate_tlu_map.py
```

Chạy A* baseline:

```powershell
python eval\astar_map_demo.py
```

Chạy Random và Greedy baseline:

```powershell
python eval\evaluate_policies.py
```

Chạy PPO evaluation nếu đã có model:

```powershell
python eval\evaluate_ppo.py
```

Xuất ảnh presentation:

```powershell
python eval\export_presentation.py
```

Sinh bảng và summary cho báo cáo:

```powershell
python eval\generate_report_summary.py
```

Chạy demo live:

```powershell
python render\live_demo.py --goal drop_t45 --policy ppo --hide-sensors
```

Nếu muốn giải thích ray-casting:

```powershell
python render\live_demo.py --goal drop_t45 --policy ppo --show-sensors
```

## 3. Các file cần biết rõ

### `map_design/generate_tlu_map.py`

Cần hiểu:

- File này là nguồn tạo bản đồ.
- Map được định nghĩa bằng hình học: `rect`, `oval`, `poly`.
- Các layer gồm đường đi, sân/nền, tòa nhà, cây, no-fly.
- POI gồm pickup, dropoff, landmark.
- Output là `.json`, `.npy` và ảnh preview.

Khi trình bày:

> Em không dùng ảnh map đơn thuần, mà chuyển khuôn viên thành semantic raster map gồm occupancy, type, height và risk để môi trường RL có thể đọc được.

### `envs/tlu_uav_env.py`

Cần hiểu:

- Đây là môi trường Gymnasium.
- UAV có state, action, reward.
- Action là 8 hướng bay.
- Collision kiểm tra theo đoạn bay, không chỉ điểm cuối.
- Reward gồm step penalty, progress reward, risk penalty, goal reward, collision penalty, timeout penalty.

Khi trình bày:

> Môi trường biến bài toán điều hướng UAV thành MDP. Agent nhận quan sát, chọn hành động, môi trường cập nhật vị trí và trả reward.

### `eval/astar_map_demo.py`

Cần hiểu:

- A* chạy trên `occupancy_grid`.
- Heuristic là khoảng cách Euclidean.
- Có kiểm tra đoạn nối giữa hai node để tránh đường đi cắt xuyên vật cản.
- Metrics gồm path length, straight distance, efficiency ratio.

Khi trình bày:

> A* được dùng như baseline quy hoạch đường đi trong bản đồ tĩnh.

### `eval/evaluate_policies.py`

Cần hiểu:

- Random: chọn hành động ngẫu nhiên.
- Greedy: chọn hành động làm giảm khoảng cách tới goal và tránh va chạm gần.
- Greedy hiện chạy 10 episode/goal để bảng baseline dễ so với PPO hơn.
- Lưu success/collision/timeout/steps/reward/final distance.

Khi trình bày:

> Random và Greedy giúp so sánh PPO với các chính sách đơn giản.

### `train/train_ppo.py`

Cần hiểu:

- Dùng Stable-Baselines3 PPO.
- `MultiGoalTluUavEnv` đổi goal qua các episode để policy học nhiều mục tiêu.
- Model được lưu vào `models/`.

Khi trình bày:

> PPO học chính sách điều khiển từ reward thay vì được lập trình đường đi cụ thể.

### `eval/evaluate_ppo.py`

Cần hiểu:

- Load model PPO.
- Chạy 10 episode cho mỗi goal.
- Lưu summary và trajectory.
- Nếu goal nào fail, giải thích là hạn chế cần fine-tune/retrain.

## 4. Các output cần dùng trong báo cáo

Ảnh map:

- `data/processed/previews/campus_render_v3.png`
- `data/processed/previews/type_map_preview.png`
- `data/processed/previews/height_map_preview.png`
- `data/processed/previews/risk_map_preview.png`

Ảnh presentation:

- `results/presentation/map_overview.png`
- `results/presentation/raster_layers.png`
- `results/presentation/baseline_comparison.png`

Bảng kết quả:

- `results/astar_metrics.csv`
- `results/policy_eval_summary.csv`
- `results/ppo_eval_summary_100k_radius30.csv`
- `results/report_summary.md`
- `results/report_tables/astar_routes.csv`
- `results/report_tables/policy_comparison.csv`
- `results/report_tables/goal_summary.csv`
- `results/report_tables/poi_validation.csv`

Ảnh trajectory:

- `results/trajectories/astar_*.png`
- `results/trajectories/random_*.png`
- `results/trajectories/greedy_*.png`
- `results/trajectories/ppo_*.png`

## 5. Checklist ôn trước khi báo cáo

### Nhóm câu hỏi về bản đồ

- Vì sao cần semantic map thay vì ảnh thường?
- `occupancy_grid` khác `risk_map` thế nào?
- Vì sao cây có thể là risk nhưng không phải lúc nào cũng là obstacle cứng?
- Vì sao xe dưới đất không nên phạt mạnh nếu UAV bay khoảng 3m?
- Làm sao kiểm tra một cửa/dropoff có bị nằm trong obstacle hay không?

### Nhóm câu hỏi về MDP

- State gồm những thành phần nào?
- Action có bao nhiêu hướng?
- Reward được tính như thế nào?
- Khi nào episode kết thúc?
- Vì sao cần ray-casting?

### Nhóm câu hỏi về thuật toán

- A* dùng để làm gì?
- Greedy khác A* thế nào?
- PPO khác A* thế nào?
- Vì sao Random gần như luôn fail?
- Vì sao Greedy có thể bị timeout?
- Vì sao A* có thể đạt 5/5 còn PPO vẫn fail ở một goal?

### Nhóm câu hỏi về kết quả

- Success rate nghĩa là gì?
- Collision rate nghĩa là gì?
- Timeout rate nghĩa là gì?
- Efficiency ratio của A* nghĩa là gì?
- Nếu PPO fail ở một mục tiêu thì giải thích thế nào?

## 6. Flow trình bày demo miệng

1. Giới thiệu bài toán: UAV giao/di chuyển trong khuôn viên trường.
2. Cho xem map overview.
3. Giải thích các lớp raster: occupancy, type, height, risk.
4. Định nghĩa MDP: state, action, transition, reward, done.
5. Nói baseline: Random, Greedy, A*.
6. Nói PPO: học policy từ reward.
7. Cho xem bảng kết quả.
8. Nói hạn chế: 2D, xe chưa dynamic, PPO cần fine-tune thêm trên map cuối.
9. Nói hướng phát triển: 3D altitude, dynamic obstacles, retraining, video demo.

## 7. Những câu nên nói trong báo cáo

Về map:

> Bản đồ được xây dựng thủ công dựa trên khu vực thực tế của trường, sau đó raster hóa thành các lớp phục vụ mô phỏng.

Về reward:

> Reward được thiết kế để khuyến khích UAV tiến gần mục tiêu, tránh va chạm và hạn chế đi qua vùng rủi ro.

Về risk:

> Risk map không nhất thiết chặn UAV, mà tạo chi phí để agent ưu tiên đường bay an toàn hơn.

Về xe:

> Do UAV được giả định bay ở độ cao khoảng 3m, xe mặt đất không được coi là vật cản cứng trong phiên bản hiện tại.

Về hạn chế:

> Kết quả PPO phụ thuộc vào bản đồ và thời gian huấn luyện; khi bản đồ thay đổi, policy cần được fine-tune để đạt hiệu quả tốt nhất.
