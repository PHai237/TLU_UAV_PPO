# Ghi chú MDP và phương pháp mô phỏng UAV

File này dùng để học và viết phần phương pháp cho báo cáo. Mỗi mục đều gắn với file code tương ứng để khi bảo vệ có thể chỉ ra dự án đang làm gì.

## 1. Bài toán đang mô phỏng

Dự án mô phỏng một UAV bay trên bản đồ 2D khuôn viên Trường Đại học Thủy Lợi. UAV xuất phát từ `Cổng sau` và cần bay tới một trong các điểm giao:

- `Cửa T45`
- `Cửa thư viện`
- `Cửa K1`
- `Cửa C1`
- `Cửa KTX số 4`

Code liên quan:

- `map_design/generate_tlu_map.py`: sinh bản đồ semantic và raster.
- `envs/tlu_uav_env.py`: định nghĩa môi trường điều khiển UAV.
- `train/train_ppo.py`: huấn luyện PPO đa mục tiêu.
- `eval/astar_map_demo.py`: baseline A*.
- `eval/evaluate_policies.py`: baseline Random và Greedy.
- `eval/evaluate_ppo.py`: đánh giá model PPO.

## 2. Các lớp bản đồ

Map không chỉ là ảnh minh họa. Nó được raster hóa thành nhiều lớp dữ liệu:

- `occupancy_grid.npy`: lưới vật cản. `1` là vùng bị chặn, `0` là vùng có thể bay qua.
- `type_map.npy`: loại semantic của từng ô, ví dụ đường đi, tòa nhà, cây xanh, sân nền, điểm giao.
- `height_map.npy`: mức cao tương đối của đối tượng.
- `risk_map.npy`: mức rủi ro từ `0` đến `1`.
- `pois.json`: tọa độ pickup/dropoff/landmark.
- `dynamic_obstacles.json`: dữ liệu xe tĩnh hiện tại, dùng làm ngữ cảnh/risk thấp.

Trong mô phỏng hiện tại, `height_map` chưa phải mô hình 3D đầy đủ. Nó là lớp xấp xỉ để phân biệt vật cản cao như tòa nhà/cây lớn với các vùng rủi ro thấp hơn. Vì UAV được giả định bay quanh độ cao khoảng 3m, xe dưới mặt đất không được xem là vật cản cứng như tòa nhà hoặc tán cây.

Ý nghĩa trong báo cáo:

- Tòa nhà và cây cao là chướng ngại quan trọng.
- Cây vừa/thấp có thể là vùng rủi ro, UAV đi qua sẽ bị phạt reward.
- Xe dưới đất không nên là vật cản cứng nếu giả định UAV bay khoảng 3m.

## 3. MDP 5 thành phần

Bài toán học tăng cường được mô hình hóa dưới dạng MDP:

```text
MDP = (S, A, P, R, gamma)
```

### S - State space

State là vector quan sát của UAV trong `envs/tlu_uav_env.py`.

Một observation gồm:

```text
[
  x_norm,
  y_norm,
  goal_dx_norm,
  goal_dy_norm,
  distance_to_goal_norm,
  local_risk,
  ray_0, ray_1, ..., ray_7
]
```

Ý nghĩa:

- `x_norm`, `y_norm`: vị trí UAV đã chuẩn hóa về khoảng gần `[-1, 1]`.
- `goal_dx_norm`, `goal_dy_norm`: vector từ UAV tới goal, giúp policy biết hướng mục tiêu.
- `distance_to_goal_norm`: khoảng cách hiện tại tới goal.
- `local_risk`: rủi ro lớn nhất quanh vị trí UAV hiện tại.
- `ray_0` đến `ray_7`: cảm biến ray-casting 8 hướng, cho biết vật cản gần hay xa.

Ghi chú về chuẩn hóa observation:

- `x_norm`, `y_norm` là vị trí tuyệt đối của UAV, được đưa về hệ tọa độ gần `[-1, 1]`.
- `goal_dx_norm`, `goal_dy_norm` là độ lệch tương đối từ UAV tới goal, chia theo kích thước map. Cách này giữ được hướng và độ xa gần của mục tiêu.
- `np.clip` được dùng để tránh giá trị vượt biên do làm tròn hoặc do UAV sát mép bản đồ.

Liên quan code:

- `_get_obs()`
- `_local_risk()`
- `_ray_cast()`

### A - Action space

Action là không gian hành động rời rạc gồm 8 hướng bay:

```text
0: lên
1: xuống
2: trái
3: phải
4: lên-phải
5: lên-trái
6: xuống-phải
7: xuống-trái
```

Các vector chéo được normalize để tốc độ theo đường chéo không nhanh hơn hướng thẳng.

Liên quan code:

- `ACTION_DIRECTIONS`
- `action_space = spaces.Discrete(8)`

### P - Transition model

Transition là cách môi trường chuyển từ state hiện tại sang state tiếp theo khi chọn action.

Trong code:

```text
new_pos = current_pos + direction * step_size
```

Sau đó môi trường kiểm tra:

- UAV có ra ngoài bản đồ không.
- Đoạn bay từ vị trí cũ tới vị trí mới có va chạm vật cản không.
- Nếu không va chạm thì cập nhật vị trí mới.

Liên quan code:

- `step()`
- `_segment_collides()`
- `_is_blocked()`

### R - Reward function

Reward hiện tại:

```text
reward = -0.10 mỗi bước
nếu va chạm:
    reward -= 100
    episode kết thúc
nếu không va chạm:
    progress = old_distance - new_distance
    reward += progress * 0.20
    reward -= local_risk * 2.0
    nếu tới goal:
        reward += 100
        episode kết thúc
nếu timeout:
    reward -= 30
```

Ý nghĩa:

- Phạt nhỏ mỗi bước để UAV không bay lòng vòng.
- Thưởng khi tiến gần goal hơn.
- Phạt khi đi vào vùng rủi ro như cụm cây.
- Phạt nặng khi đâm vào vật cản.
- Thưởng lớn khi tới đích.
- Phạt timeout nếu bay quá lâu mà không tới đích.

Liên quan code:

- `step()` trong `envs/tlu_uav_env.py`.

### gamma - Discount factor

`gamma` nằm trong cấu hình PPO ở `train/train_ppo.py`.

Ý nghĩa:

- `gamma` càng gần `1`, agent càng quan tâm tới reward dài hạn.
- Trong bài toán đường đi, điều này giúp agent không chỉ nhìn lợi ích trước mắt mà còn học cách đi vòng để tránh vật cản.

## 4. Thuật toán A*

A* là baseline quy hoạch đường đi trên `occupancy_grid`.

Ý tưởng:

- Duyệt các ô có thể đi.
- Dùng heuristic là khoảng cách Euclidean tới goal.
- Tìm đường ngắn hợp lệ từ pickup tới dropoff.

Vì map lớn `1400 x 900`, A* dùng bước nhảy `step=8` để chạy nhanh hơn.

Liên quan code:

- `eval/astar_map_demo.py`
- `astar()`
- `heuristic()`
- `is_segment_free()`

Vai trò trong báo cáo:

- A* chứng minh bản đồ có đường đi hợp lệ.
- A* là baseline mạnh cho map tĩnh.
- PPO khác A* ở chỗ PPO học policy từ tương tác và reward, không trực tiếp tìm đường bằng thuật toán search.

## 5. Random, Greedy và PPO

### Random

Random chọn hành động ngẫu nhiên.

Mục đích:

- Là baseline yếu.
- Cho thấy nếu UAV không có chiến lược thì dễ va chạm.

### Greedy

Greedy chọn hành động làm giảm khoảng cách tới goal và tránh va chạm trước mắt.

Mục đích:

- Là baseline đơn giản nhưng hợp lý hơn Random.
- Có thể thành công ở các goal dễ.
- Có thể timeout ở các goal cần đi vòng hoặc có vùng risk.

### PPO

PPO là thuật toán policy gradient ổn định, học policy bằng cách tương tác với môi trường.

Mục đích:

- Học cách cân bằng giữa tiến gần goal, tránh vật cản và tránh risk.
- Phù hợp để trình bày học tăng cường trong môi trường mô phỏng UAV.

Liên quan code:

- `train/train_ppo.py`
- `eval/evaluate_ppo.py`

## 6. Cách giải thích hạn chế

Nếu PPO chưa thành công ở toàn bộ goal, có thể trình bày là hạn chế thực nghiệm:

- Map đã được chỉnh nhiều lần nên model cũ có thể chưa tối ưu cho bản đồ cuối.
- Một số khu vực có cây/risk tạo thử thách cao hơn.
- Cần fine-tune hoặc train thêm để tăng success rate.
- Hiện môi trường mới mô phỏng 2D, chưa mô phỏng đầy đủ động lực học UAV 3D.

Đây là điểm hợp lý để viết phần hướng phát triển:

- Thêm altitude control.
- Thêm dynamic risk layer.
- Fine-tune PPO trên map cuối.
- Xuất video/demo trực quan hơn.
