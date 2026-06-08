# TLU UAV PPO - Tom tat ket qua cho bao cao

File nay duoc sinh tu `eval/generate_report_summary.py` dua tren cac file trong `results/`.

## Diem chinh

- Ban do hien tai co 5 diem giao: T45, Thu vien, K1, C1, KTX so 4.
- A* tim duoc duong cho 5/5 muc tieu tren occupancy grid.
- PPO 100k tren ban do hien tai dat trung binh 80.0% success neu tinh theo 5 muc tieu.
- Random baseline dung de chung minh hanh dong ngau nhien de va cham.
- Greedy baseline dung de chung minh chien luoc tham lam co the thanh cong o mot so diem, nhung de timeout o cac diem can di vong.

## Bang A* Baseline

| goal | path_found | path_points | path_length_px | straight_distance_px | efficiency_ratio |
| --- | --- | --- | --- | --- | --- |
| Hoi truong T45 | True | 107 | 950.59 | 884.6 | 1.0746 |
| Thu vien | True | 70 | 617.26 | 502.24 | 1.229 |
| K1 | True | 54 | 466.77 | 330.62 | 1.4118 |
| C1 | True | 80 | 652.27 | 636.52 | 1.0247 |
| KTX so 4 | True | 49 | 483.21 | 452.07 | 1.0689 |

## Bang So Sanh Policy

| policy | goal | episodes | success_rate | collision_rate | timeout_rate | avg_steps | avg_reward | avg_final_distance_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random baseline | Hoi truong T45 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.25 | 887.41 |
| Random baseline | Thu vien | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.81 | 507.83 |
| Random baseline | K1 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -139.37 | 339.0 |
| Random baseline | C1 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.39 | 640.03 |
| Random baseline | KTX so 4 | 20 | 0.0% | 100.0% | 0.0% | 145.55 | -138.46 | 455.93 |
| Greedy baseline | Hoi truong T45 | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -869.57 | 393.44 |
| Greedy baseline | Thu vien | 10 | 100.0% | 0.0% | 0.0% | 74.0 | 118.24 | 24.06 |
| Greedy baseline | K1 | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -1028.97 | 165.47 |
| Greedy baseline | C1 | 10 | 100.0% | 0.0% | 0.0% | 78.0 | 206.6 | 27.0 |
| Greedy baseline | KTX so 4 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 151.39 | 26.01 |
| PPO 100k radius30 | Hoi truong T45 | 10 | 100.0% | 0.0% | 0.0% | 116.0 | 218.51 | 29.06 |
| PPO 100k radius30 | Thu vien | 10 | 0.0% | 0.0% | 100.0% | 500.0 | -83.06 | 507.54 |
| PPO 100k radius30 | K1 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 152.8 | 28.1 |
| PPO 100k radius30 | C1 | 10 | 100.0% | 0.0% | 0.0% | 78.0 | 208.6 | 29.53 |
| PPO 100k radius30 | KTX so 4 | 10 | 100.0% | 0.0% | 0.0% | 57.0 | 154.41 | 29.41 |

## Policy Tot Nhat Theo Tung Muc Tieu

| goal | best_policy | best_success_rate | best_avg_steps | best_avg_reward |
| --- | --- | --- | --- | --- |
| Hoi truong T45 | PPO 100k radius30 | 100.0% | 116.0 | 218.51 |
| Thu vien | Greedy baseline | 100.0% | 74.0 | 118.24 |
| K1 | PPO 100k radius30 | 100.0% | 57.0 | 152.8 |
| C1 | PPO 100k radius30 | 100.0% | 78.0 | 208.6 |
| KTX so 4 | PPO 100k radius30 | 100.0% | 57.0 | 154.41 |

## Kiem Tra Pickup/Dropoff

Bang nay xac nhan cac diem pickup/dropoff nam trong ban do va khong nam tren obstacle.

| group | id | label | x | y | inside_map | occupancy_at_point | is_free | nearest_obstacle_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pickup_points | pickup_back_gate | Cổng sau | 1350 | 490 | True | 0 | True | 66.31 |
| dropoff_points | drop_t45 | Cửa T45 | 500 | 245 | True | 0 | True | 26.0 |
| dropoff_points | drop_library | Cửa thư viện | 985 | 145 | True | 0 | True | 26.0 |
| dropoff_points | drop_k1 | Cửa K1 | 1105 | 268 | True | 0 | True | 25.0 |
| dropoff_points | drop_c1 | Cửa C1 | 715 | 446 | True | 0 | True | 17.0 |
| dropoff_points | drop_dorm4 | Cửa KTX số 4 | 962 | 722 | True | 0 | True | 18.0 |

## Cach doc ket qua

- `success_rate`: ty le episode ket thuc bang viec UAV vao vung goal.
- `collision_rate`: ty le episode va cham voi obstacle/no-fly.
- `timeout_rate`: ty le episode het `max_steps` nhung chua toi goal.
- `avg_steps`: so buoc trung binh; cang thap cang tot neu van thanh cong.
- `avg_reward`: tong reward trung binh; cao hon thuong la policy tot hon.
- `avg_final_distance_px`: khoang cach con lai toi goal khi episode ket thuc.
- `efficiency_ratio`: do dai duong A* / khoang cach thang; gan 1 nghia la duong gan toi uu ve hinh hoc.

## Goi y viet bao cao

- Dung A* lam baseline quy hoach duong di tren ban do tinh.
- Dung Random va Greedy lam baseline chinh sach don gian.
- Dung PPO de trinh bay huong hoc tang cuong: policy hoc tu reward thay vi duoc lap trinh quy tac duong di.
- Neu PPO that bai o mot goal, trinh bay nhu han che thuc nghiem va ly do can fine-tune/retrain tren ban do cuoi.
