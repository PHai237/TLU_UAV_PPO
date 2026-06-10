# Mã giả và đối chiếu với code

File này dùng để kiểm tra các thuật toán trong đồ án có được triển khai đúng theo mã giả hay không, đồng thời cung cấp nội dung có thể đưa vào báo cáo.

## 1. Kết luận nhanh

- Môi trường UAV triển khai đúng chu trình tương tác MDP/Gymnasium.
- A* triển khai đúng cấu trúc thuật toán A* chuẩn, có bổ sung kiểm tra đoạn nối để phù hợp bản đồ raster.
- Random và Greedy là các baseline đơn giản, được triển khai đúng mục đích so sánh.
- PPO sử dụng implementation chuẩn từ thư viện Stable-Baselines3. Code dự án chịu trách nhiệm xây dựng môi trường, cấu hình siêu tham số và gọi quá trình học.

## 2. Chu trình môi trường MDP

### Mã giả

```text
Khởi tạo môi trường và trạng thái ban đầu s

Lặp cho tới khi episode kết thúc:
    Agent chọn hành động a từ trạng thái s
    Tính vị trí mới dự kiến
    Kiểm tra va chạm

    Nếu va chạm:
        phạt collision
        kết thúc episode
    Ngược lại:
        cập nhật vị trí
        tính progress tới mục tiêu
        cộng progress reward
        trừ risk penalty

        nếu vào vùng mục tiêu:
            thưởng goal
            kết thúc episode

    nếu vượt max_steps:
        phạt timeout
        kết thúc episode

    trả về trạng thái mới, reward, terminated, truncated, info
```

### Đối chiếu code

- File: `envs/tlu_uav_env.py`
- Hàm chính: `reset()`, `step()`
- Va chạm: `_segment_collides()`
- Observation: `_get_obs()`
- Risk: `_local_risk()`

Đây là cách triển khai phù hợp API Gymnasium và mô hình MDP của đồ án.

## 3. Thuật toán A*

### Mã giả chuẩn

```text
OPEN = hàng đợi ưu tiên chứa start
g(start) = 0

Trong khi OPEN không rỗng:
    current = node có f nhỏ nhất trong OPEN

    nếu current đủ gần goal và nối thẳng tới goal không va chạm:
        lưu parent của goal
        trả về đường đi được truy vết từ goal

    đánh dấu current đã xét

    với mỗi neighbor của current:
        nếu đoạn current -> neighbor va chạm:
            bỏ qua

        tentative_g = g(current) + cost(current, neighbor)

        nếu tentative_g tốt hơn g(neighbor):
            parent(neighbor) = current
            g(neighbor) = tentative_g
            f(neighbor) = g(neighbor) + h(neighbor, goal)
            thêm neighbor vào OPEN

Nếu OPEN rỗng:
    không tìm được đường
```

### Đối chiếu code

- File: `eval/astar_map_demo.py`
- `open_set`: hàng đợi ưu tiên OPEN.
- `g_score`: chi phí thực từ start.
- `heuristic()`: khoảng cách Euclidean tới goal.
- `came_from`: lưu node cha để dựng lại đường.
- `is_segment_free()`: kiểm tra cạnh nối không xuyên obstacle.
- `reconstruct_path()`: truy vết đường đi.

### Biến thể của đồ án

- A* di chuyển theo 8 hướng.
- `step=8` được dùng để giảm số node cần xét trên map `1400 x 900`.
- Goal được chấp nhận khi node hiện tại đủ gần và đoạn nối cuối tới goal không va chạm.

Đây vẫn là A*, nhưng chạy trên lưới thưa hơn để phù hợp kích thước bản đồ.

## 4. Random baseline

### Mã giả

```text
Trong khi episode chưa kết thúc:
    chọn ngẫu nhiên một hành động trong 8 hướng
    thực hiện hành động trong môi trường
```

### Đối chiếu code

- `envs/tlu_uav_env.py`: `sample_random_action()`
- `eval/evaluate_policies.py`: `run_episode(policy_name="random", ...)`

Random không phải thuật toán tìm đường tối ưu; nó là baseline yếu để chứng minh bài toán cần chiến lược điều hướng.

## 5. Greedy baseline

### Mã giả

```text
Với mỗi hành động có thể:
    tính vị trí candidate
    nếu candidate gây va chạm:
        bỏ qua

    score = khoảng cách candidate tới goal + risk penalty

Chọn hành động có score nhỏ nhất

Nếu không có hành động an toàn:
    chọn fallback ngẫu nhiên
```

### Đối chiếu code

- File: `envs/tlu_uav_env.py`
- Hàm: `greedy_action()`

Greedy chỉ tối ưu một bước trước mắt, không có OPEN/CLOSED hoặc bộ nhớ đường đi như A*. Vì vậy Greedy có thể bị kẹt khi cần tạm thời đi xa goal để vòng qua vật cản.

## 6. Thuật toán PPO

### Mã giả PPO rút gọn

```text
Khởi tạo policy pi_theta và value function V_phi

Lặp cho tới khi đủ total_timesteps:
    Thu thập rollout bằng policy hiện tại:
        quan sát state
        lấy action từ policy
        thực hiện action trong environment
        lưu state, action, reward, done, value, log_probability

    Tính return và advantage bằng GAE

    Lặp qua nhiều epoch và mini-batch:
        tính probability ratio giữa policy mới và policy cũ
        tính clipped surrogate objective
        tính value loss
        tính entropy bonus
        cập nhật tham số bằng gradient descent

Lưu model đã học
```

### Đối chiếu code

- File cấu hình/train: `train/train_ppo.py`
- Môi trường cung cấp rollout: `envs/tlu_uav_env.py`
- Implementation PPO: `stable_baselines3.PPO`
- Bắt đầu học: `model.learn(total_timesteps=100_000)`
- Đánh giá model: `eval/evaluate_ppo.py`

Code dự án không tự viết nội bộ PPO vì sử dụng implementation đã được kiểm thử của Stable-Baselines3. Các tham số chính đang dùng:

```text
learning_rate = 3e-4
n_steps = 512
batch_size = 64
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

## 7. Cách trình bày khi bảo vệ

> Đồ án tự xây dựng môi trường MDP, observation, action, transition và reward. Thuật toán A* và các baseline được cài đặt trực tiếp trong dự án. Riêng PPO sử dụng implementation chuẩn từ Stable-Baselines3; đồ án cấu hình PPO và cung cấp môi trường Gymnasium để thuật toán thu thập rollout và học policy.

Không nên nói rằng đồ án tự cài đặt toàn bộ PPO từ công thức toán học, vì phần tối ưu PPO do Stable-Baselines3 thực hiện.
