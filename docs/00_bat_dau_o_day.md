# Bắt đầu ở đây

File này là lộ trình đọc nhanh để hiểu dự án trước khi viết báo cáo hoặc demo với thầy.

## 1. Dự án này đang làm gì?

Dự án mô phỏng một UAV bay trong khuôn viên Trường Đại học Thủy Lợi trên bản đồ semantic 2D. UAV xuất phát từ `Cổng sau` và cần đi tới các điểm như `Hội trường T45`, `Thư viện`, `K1`, `C1`, `KTX số 4`.

Bài toán được mô hình hóa thành MDP và giải bằng PPO. Các phương pháp Random, Greedy và A* được dùng làm baseline để so sánh.

## 2. Thứ tự học nên theo

1. Đọc file này trước.
2. Đọc `docs/01_flow_chay_code_va_demo.md` để biết chạy lệnh nào và file nào sinh ra output nào.
3. Đọc `docs/02_mdp_va_thuat_toan.md` để hiểu MDP, state, action, reward, A*, PPO.
4. Đọc `docs/03_ket_qua_hien_tai.md` để biết kết quả hiện tại và cách giải thích PPO fail ở Thư viện.
5. Đọc `docs/04_file_output_can_dung.md` để biết file nào dùng cho báo cáo.
6. Đọc `docs/05_ghi_chu_ban_do.md` nếu cần giải thích giả định thiết kế map.
7. Đọc `docs/06_ma_gia_va_doi_chieu_code.md` khi viết phần mã giả hoặc cần đối chiếu thuật toán với code.

## 3. Flow triển khai code

```text
1. Sinh bản đồ
   map_design/generate_tlu_map.py
   -> semantic_map.json, pois.json
   -> occupancy_grid.npy, type_map.npy, height_map.npy, risk_map.npy
   -> ảnh preview map

2. Tạo môi trường UAV
   envs/tlu_uav_env.py
   -> đọc map raster
   -> định nghĩa state, action, transition, reward, done

3. Train hoặc load PPO
   train/train_ppo.py
   -> train model PPO đa mục tiêu
   -> lưu models/ppo_tlu_uav_100k_radius30.zip

4. Đánh giá baseline
   eval/astar_map_demo.py
   eval/evaluate_policies.py
   -> A*, Random, Greedy

5. Đánh giá PPO
   eval/evaluate_ppo.py
   -> PPO trên 5 mục tiêu
   -> bảng kết quả và ảnh trajectory

6. Xuất tài liệu báo cáo
   eval/export_presentation.py
   eval/generate_report_summary.py
   -> ảnh đẹp, bảng CSV, report_summary.md

7. Demo trực quan
   render/live_demo.py
   -> chạy mô phỏng live theo policy đã chọn
```

## 4. Bạn cần hiểu gì để bảo vệ?

### Bản đồ

- Vì sao dùng semantic map thay vì ảnh thường.
- `occupancy_grid`: vùng bị chặn hay đi được.
- `type_map`: loại đối tượng như đường, tòa nhà, cây, sân, pickup/dropoff.
- `height_map`: độ cao tương đối, dùng để mô phỏng vật cản cao/thấp.
- `risk_map`: vùng rủi ro, đi qua bị phạt reward nhưng không nhất thiết là obstacle cứng.

### MDP

- `S` là state/observation của UAV.
- `A` là 8 hướng bay rời rạc.
- `P` là transition: vị trí mới bằng vị trí cũ cộng vector hành động.
- `R` là reward: thưởng tiến gần goal, phạt va chạm, phạt risk, phạt timeout.
- `gamma` là hệ số chiết khấu reward dài hạn trong PPO.

### Thuật toán

- Random: baseline yếu, hành động ngẫu nhiên.
- Greedy: đi theo hướng giảm khoảng cách tới goal, dễ kẹt nếu cần đi vòng.
- A*: baseline quy hoạch đường đi trên bản đồ tĩnh, biết toàn bộ map.
- PPO: policy học qua tương tác và reward, chỉ nhận observation của môi trường.

### Kết quả

- A* đạt 5/5, chứng minh map có đường đi hợp lệ.
- PPO hiện đạt 4/5, không va chạm, fail ở Thư viện do timeout.
- PPO fail Thư viện là hạn chế thực nghiệm hợp lý: model 100k chưa hội tụ đủ trên map cuối.
- Xe động chưa triển khai thành obstacle cứng; với giả định UAV bay khoảng 3m, xe mặt đất chỉ là ngữ cảnh/risk thấp.

## 5. Tên file code đọc như thế nào?

| File | Hiểu đơn giản là |
| --- | --- |
| `map_design/generate_tlu_map.py` | Bộ dựng bản đồ trường và sinh các lớp raster |
| `envs/tlu_uav_env.py` | Môi trường UAV/MDP để agent tương tác |
| `train/train_ppo.py` | Script train PPO từ đầu |
| `train/fine_tune_ppo.py` | Script train tiếp từ model PPO đã có |
| `eval/astar_map_demo.py` | Chạy A* để kiểm tra map có đường đi không |
| `eval/evaluate_policies.py` | Chạy Random và Greedy baseline |
| `eval/evaluate_ppo.py` | Chạy PPO trên 5 mục tiêu và xuất trajectory |
| `eval/export_presentation.py` | Xuất ảnh đẹp cho báo cáo/slide |
| `eval/generate_report_summary.py` | Gom kết quả thành bảng và summary cho báo cáo |
| `render/live_demo.py` | Demo trực quan đường bay của UAV |

## 6. Lệnh tối thiểu cần nhớ

```powershell
python map_design\generate_tlu_map.py
python eval\astar_map_demo.py
python eval\evaluate_policies.py
python eval\evaluate_ppo.py
python eval\export_presentation.py
python eval\generate_report_summary.py
```

Demo live gọn:

```powershell
python render\live_demo.py --goal drop_t45 --policy ppo --hide-sensors
```

Demo có ray-casting nếu thầy hỏi cảm biến:

```powershell
python render\live_demo.py --goal drop_t45 --policy ppo --show-sensors
```

## 7. Câu chốt để nói trong báo cáo

> Đồ án xây dựng môi trường mô phỏng UAV trên bản đồ semantic 2D của khuôn viên trường, mô hình hóa bài toán điều hướng thành MDP và áp dụng PPO để học chính sách điều khiển. Các baseline Random, Greedy và A* được dùng để đánh giá tương đối hiệu quả của PPO.
