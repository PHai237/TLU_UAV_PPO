# Kết quả hiện tại

File này ghi lại trạng thái kết quả sau khi chốt bản đồ hiện tại. Bảng tự động, chi tiết hơn nằm ở `results/report_summary.md` và `results/report_tables/`.

## 1. Trạng thái bản đồ

- Bản đồ semantic 2D đã chốt cho khu vực Cổng sau, A1, T45, Thư viện, K1, C1, B4 và KTX số 4.
- Các lớp map hiện hành gồm `occupancy_grid.npy`, `type_map.npy`, `height_map.npy`, `risk_map.npy`.
- Các điểm pickup/dropoff đều nằm trong map và không nằm trên obstacle.
- `Cửa thư viện` hiện ở `(985, 145)`, `occupancy_at_point = 0`, tức là hợp lệ.

## 2. Kết quả A*

A* tìm được đường tới đủ 5/5 mục tiêu.

| Mục tiêu | Path found | Path length px | Efficiency ratio |
| --- | --- | ---: | ---: |
| Hội trường T45 | Có | 950.59 | 1.0746 |
| Thư viện | Có | 617.26 | 1.2290 |
| K1 | Có | 466.77 | 1.4118 |
| C1 | Có | 652.27 | 1.0247 |
| KTX số 4 | Có | 483.21 | 1.0689 |

Ý nghĩa: bản đồ có tính liên thông; A* đóng vai trò baseline quy hoạch đường đi trên map tĩnh.

## 3. Kết quả Random và Greedy

Random baseline:

- Chạy 20 episode cho mỗi mục tiêu.
- Tỷ lệ thành công: 0% cho toàn bộ mục tiêu.
- Tỷ lệ va chạm: 100% cho toàn bộ mục tiêu.
- Vai trò: chứng minh hành động ngẫu nhiên không đủ cho bài toán điều hướng có vật cản.

Greedy baseline:

- Chạy 10 episode cho mỗi mục tiêu.
- Thành công ở Thư viện, C1 và KTX số 4.
- Timeout ở T45 và K1.
- Vai trò: chứng minh chính sách tham lam có thể tốt ở một số tuyến nhưng dễ kẹt khi cần đi vòng.

| Mục tiêu | Success rate | Timeout rate | Avg steps | Avg reward |
| --- | ---: | ---: | ---: | ---: |
| Hội trường T45 | 0% | 100% | 500.0 | -869.57 |
| Thư viện | 100% | 0% | 74.0 | 118.24 |
| K1 | 0% | 100% | 500.0 | -1028.97 |
| C1 | 100% | 0% | 78.0 | 206.60 |
| KTX số 4 | 100% | 0% | 57.0 | 151.39 |

## 4. Kết quả PPO 100k radius30

PPO được đánh giá lại trên bản đồ hiện tại.

| Mục tiêu | Success rate | Collision rate | Timeout rate | Avg steps | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hội trường T45 | 100% | 0% | 0% | 116.0 | 218.51 |
| Thư viện | 0% | 0% | 100% | 500.0 | -83.06 |
| K1 | 100% | 0% | 0% | 57.0 | 152.80 |
| C1 | 100% | 0% | 0% | 78.0 | 208.60 |
| KTX số 4 | 100% | 0% | 0% | 57.0 | 154.41 |

Tóm tắt:

- PPO đạt thành công 4/5 mục tiêu.
- Không có collision trong PPO evaluation.
- Điểm yếu hiện tại là Thư viện: PPO timeout và còn cách mục tiêu khoảng 507.54 px.
- Đây là hạn chế hợp lý để đưa vào báo cáo: model cần fine-tune/retrain thêm trên bản đồ cuối.
- Đã thử fine-tune thêm từ model 100k, nhưng candidate chưa đạt 5/5 và có trường hợp làm giảm kết quả ở mục tiêu khác. Vì vậy model chính vẫn giữ là `ppo_tlu_uav_100k_radius30.zip`.

## 5. Nhận xét để viết báo cáo

- A* được dùng để xác nhận map có đường đi hợp lệ.
- Random thể hiện baseline yếu.
- Greedy thể hiện baseline heuristic có thể fail ở tuyến cần đi vòng.
- PPO học được chính sách tốt hơn ở 4/5 mục tiêu nhưng chưa hội tụ hoàn toàn.
- Vì bản đồ đã thay đổi nhiều sau quá trình chỉnh sửa, việc PPO fail ở Thư viện có thể giải thích là do model 100k chưa được fine-tune đủ trên bản đồ cuối.

## 6. File kết quả nên dùng

- `results/report_summary.md`
- `results/report_tables/astar_routes.csv`
- `results/report_tables/policy_comparison.csv`
- `results/report_tables/goal_summary.csv`
- `results/report_tables/poi_validation.csv`
- `results/presentation/map_overview.png`
- `results/presentation/raster_layers.png`
- `results/presentation/baseline_comparison.png`
