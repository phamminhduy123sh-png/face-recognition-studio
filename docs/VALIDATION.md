# Kiểm chứng bản phân phối

Ngày chạy: **13/11/2026**. Đây là kiểm thử phần mềm, không phải báo cáo accuracy trên ba người thật.

## Đã chạy

- `python -m pytest -q`: **67 tests đạt**, cùng **34 subtests đạt**.
- Nạp hai state_dict người dùng cung cấp bằng `weights_only=True`; đối chiếu đầy đủ tên và shape tham số.
- Chạy một epoch train/prototype/validation/lưu bundle/evaluate bằng **ảnh tổng hợp**.
- Warm-start FaceCNN từ trọng số gốc qua một epoch tổng hợp; không diễn giải kết quả thành nhận diện thật.
- Tải trọng số ResNet18 ImageNet chính thức của torchvision và chạy forward 224×224: output 128 chiều hữu hạn.
- Xây dựng wheel Python thành công; tài nguyên giao diện được đóng gói trong package.
- Chromium headless, 1440×1000 và 390×844: năm tab hoạt động, 18 ô phiên, không lỗi JavaScript và không tràn ngang mobile.
- Upload một ảnh chân dung kiểm thử công khai của Pillow: detector thật tìm được mặt, backbone gốc sinh cosine hữu hạn;
  ứng dụng trả `accepted=false`, `uncalibrated`, chỉ hiện ứng viên. Ảnh kiểm thử không được gán vào dataset hoặc đưa vào ZIP.
- Luồng camera trình duyệt kiểm thử bằng MediaStream từ canvas: snapshot chạy qua API; lưu ảnh không có mặt bị từ chối.
- Các API kiểm tra ảnh lỗi, alpha/EXIF, giới hạn upload, ba người/sáu phiên, job nền và checkpoint legacy.
- API train chỉ cho phép một tác vụ; đánh giá/hiệu chỉnh dùng đúng checkpoint đang được chọn.
- Giải nén ZIP vào thư mục mới, chạy `init`, nạp legacy, phục vụ UI/API và chạy lại bộ kiểm thử thành công.
- Nút tải ZIP chạy qua trình duyệt ở 1280×720; nội dung tải khớp byte với archive đã tạo.
- Train lại lưu artifacts cũ vào archive; không tái dùng ngưỡng hoặc kết quả test của model trước.
- Ngưỡng gắn hash model và preprocessing; kiểm tra các biên cosine/nextafter và quyết định reject nhất quán.

Ảnh giao diện: [Desktop](screenshots/desktop.png), [18 phiên](screenshots/sessions.png), [Mobile](screenshots/mobile.png).

Bổ sung sửa khởi động: trình duyệt chỉ mở sau khi Uvicorn lắng nghe (kiểm thử HTTP200 thật); lỗi khởi động Windows giữ lại cửa sổ thông báo. Script Windows vẫn cần kiểm tra trên máy người dùng.

## Môi trường kiểm thử

| Thành phần | Phiên bản |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.15.0+cpu |
| torchvision | 0.29.2+cpu |
| OpenCV | 4.14.0 |
| FastAPI | 0.141.1 |
| NumPy | 2.3.5 |
| Pillow | 13.0.0 |

Pytest có hai cảnh báo deprecation từ plugin AnyIO và TestClient của thư viện ngoài;
không có lỗi kiểm thử. requirements cho phép các phiên bản phù hợp khác; không khẳng định
đã kiểm tra mọi tổ hợp Python, hệ điều hành hoặc CUDA.

## Chưa thể xác nhận từ các file đầu vào hiện tại

- Chưa có ảnh Dũng/Duy/Hưng theo sáu phiên để đo accuracy, macro-F1, confusion matrix hoặc chọn mô hình tốt hơn.
- Chưa có người lạ validation/test để đo FAR/FRR thực tế; không cung cấp ngưỡng mặc định giả là đã hiệu chỉnh.
- Chưa thử webcam vật lý của người dùng, GPU CUDA, hoặc script khởi động trên Windows/macOS.
- ImageNet là pretrained phân loại ảnh, không phải một bộ trọng số pretrained nhận diện mặt chuyên dụng.
  ResNet18 cần so sánh với FaceCNN trên validation thật trước khi kết luận mô hình nào tốt hơn.
- Haar có hạn chế với góc nghiêng, che mặt, mặt nhỏ; không có liveness/anti-spoof.

Sau khi thu đủ dữ liệu: audit → train/chọn checkpoint bằng validation → hiệu chỉnh ngưỡng bằng
known/unknown validation → chốt cấu hình → evaluate test. Không chọn tham số theo kết quả test.
