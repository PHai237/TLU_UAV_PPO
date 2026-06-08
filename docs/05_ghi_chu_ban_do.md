# Ghi chú thiết kế bản đồ

Phiên bản bản đồ hiện tại tập trung vào việc tạo một môi trường mô phỏng 2D đủ sạch để phục vụ huấn luyện và đánh giá PPO.

Bản đồ chưa nhằm mục tiêu tái tạo chính xác GPS của khuôn viên Trường Đại học Thủy Lợi. Thay vào đó, bản đồ được thiết kế theo hướng semantic map, trong đó mỗi vùng được gán nhãn như đường đi, tòa nhà, sân, cây xanh, vùng cấm bay, pickup và dropoff.

Các lớp dữ liệu chính gồm:

- `occupancy_grid.npy`: xác định vùng có thể đi qua và vùng bị chặn.
- `type_map.npy`: lưu loại đối tượng ngữ nghĩa tại từng pixel.
- `height_map.npy`: biểu diễn mức độ cao của vật cản.
- `risk_map.npy`: biểu diễn mức độ rủi ro khi UAV bay qua từng vùng.

Ở phiên bản hiện tại, phương tiện được xem là vật thể tĩnh và chỉ ảnh hưởng tới risk/height map. Vật cản động sẽ được phát triển ở các phiên bản sau.
