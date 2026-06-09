# TLU UAV PPO - Tóm tắt kết quả cho báo cáo

File này được sinh từ `eval/generate_report_summary.py` dựa trên các file trong `results/`.

## Điểm chính

- Bản đồ hiện tại có 5 điểm giao: T45, Thư viện, K1, C1, KTX số 4.
- A* tìm được đường cho 5/5 mục tiêu trên occupancy grid.
- PPO 100k trên bản đồ hiện tại đạt trung bình 80.0% success nếu tính theo 5 mục tiêu.
- Random baseline dùng để chứng minh hành động ngẫu nhiên dễ va chạm.
- Greedy baseline dùng để chứng minh chiến lược tham lam có thể thành công ở một số điểm, nhưng dễ timeout ở các điểm cần đi vòng.
- PPO evaluation dùng `deterministic=True` với điểm xuất phát cố định; các episode cùng goal kiểm tra tính nhất quán của policy.

## Bảng A* Baseline

| Mục tiêu | Tìm được đường | Số điểm đường đi | Độ dài đường đi (px) | Khoảng cách thẳng (px) | Tỷ lệ hiệu quả |
| --- | --- | --- | --- | --- | --- |
| Hội trường T45 | Có | 107 | 950.59 | 884.6 | 1.0746 |
| Thư viện | Có | 70 | 617.26 | 502.24 | 1.229 |
| K1 | Có | 54 | 466.77 | 330.62 | 1.4118 |
| C1 | Có | 80 | 652.27 | 636.52 | 1.0247 |
| KTX số 4 | Có | 49 | 483.21 | 452.07 | 1.0689 |

## Bảng so sánh policy

| Chính sách | Mục tiêu | Số episode | Tỷ lệ thành công | Tỷ lệ va chạm | Tỷ lệ quá thời gian | Số bước TB | Reward TB | Khoảng cách cuối TB (px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random baseline | Hội trường T45 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.25 | 887.41 |
| Random baseline | Thư viện | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.81 | 507.83 |
| Random baseline | K1 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -139.37 | 339.0 |
| Random baseline | C1 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.39 | 640.03 |
| Random baseline | KTX số 4 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.46 | 455.93 |
| Greedy baseline | Hội trường T45 | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -869.57 | 393.44 |
| Greedy baseline | Thư viện | 10 | 100.0% | 0.0% | 0.0% | 74.0 | 118.24 | 24.06 |
| Greedy baseline | K1 | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -1028.97 | 165.47 |
| Greedy baseline | C1 | 10 | 100.0% | 0.0% | 0.0% | 78.0 | 206.6 | 27.0 |
| Greedy baseline | KTX số 4 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 151.39 | 26.01 |
| PPO 100k bán kính 30 | Hội trường T45 | 10 | 100.0% | 0.0% | 0.0% | 116.0 | 218.51 | 29.06 |
| PPO 100k bán kính 30 | Thư viện | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -83.06 | 507.54 |
| PPO 100k bán kính 30 | K1 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 152.8 | 28.1 |
| PPO 100k bán kính 30 | C1 | 10 | 100.0% | 0.0% | 0.0% | 78.0 | 208.6 | 29.53 |
| PPO 100k bán kính 30 | KTX số 4 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 154.41 | 29.41 |

## Policy tốt nhất theo từng mục tiêu

| Mục tiêu | Chính sách tốt nhất | Tỷ lệ thành công tốt nhất | Số bước TB tốt nhất | Reward TB tốt nhất |
| --- | --- | --- | --- | --- |
| Hội trường T45 | PPO 100k bán kính 30 | 100.0% | 116.0 | 218.51 |
| Thư viện | Greedy baseline | 100.0% | 74.0 | 118.24 |
| K1 | PPO 100k bán kính 30 | 100.0% | 57.0 | 152.8 |
| C1 | PPO 100k bán kính 30 | 100.0% | 78.0 | 208.6 |
| KTX số 4 | PPO 100k bán kính 30 | 100.0% | 57.0 | 154.41 |

## Kiểm tra pickup/dropoff

Bảng này xác nhận các điểm pickup/dropoff nằm trong bản đồ và không nằm trên obstacle.

| Nhóm | Mã điểm | Tên hiển thị | x | y | Trong bản đồ | Obstacle tại điểm | Điểm hợp lệ | Obstacle gần nhất (px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pickup_points | pickup_back_gate | Cổng sau | 1350 | 490 | Có | 0 | Có | 66.31 |
| dropoff_points | drop_t45 | Cửa T45 | 500 | 245 | Có | 0 | Có | 26.0 |
| dropoff_points | drop_library | Cửa thư viện | 985 | 145 | Có | 0 | Có | 26.0 |
| dropoff_points | drop_k1 | Cửa K1 | 1105 | 268 | Có | 0 | Có | 25.0 |
| dropoff_points | drop_c1 | Cửa C1 | 715 | 446 | Có | 0 | Có | 17.0 |
| dropoff_points | drop_dorm4 | Cửa KTX số 4 | 962 | 722 | Có | 0 | Có | 18.0 |

## Cách đọc kết quả

- `success_rate`: tỷ lệ episode kết thúc bằng việc UAV vào vùng goal.
- `collision_rate`: tỷ lệ episode va chạm với obstacle/no-fly.
- `timeout_rate`: tỷ lệ episode hết `max_steps` nhưng chưa tới goal.
- `avg_steps`: số bước trung bình; càng thấp càng tốt nếu vẫn thành công.
- `avg_reward`: tổng reward trung bình; cao hơn thường là policy tốt hơn.
- `avg_final_distance_px`: khoảng cách còn lại tới goal khi episode kết thúc.
- `efficiency_ratio`: độ dài đường A* / khoảng cách thẳng; gần 1 nghĩa là đường gần tối ưu về hình học.

## Gợi ý viết báo cáo

- Dùng A* làm baseline quy hoạch đường đi trên bản đồ tĩnh.
- Dùng Random và Greedy làm baseline chính sách đơn giản.
- Dùng PPO để trình bày hướng học tăng cường: policy học từ reward thay vì được lập trình quy tắc đường đi.
- Nếu PPO thất bại ở một goal, trình bày như hạn chế thực nghiệm và lý do cần fine-tune/retrain trên bản đồ cuối.
- Với Thư viện, quỹ đạo kẹt gần biên trên cho thấy policy chưa học được chiến lược tiếp cận goal; không phải do goal nằm trên obstacle.
