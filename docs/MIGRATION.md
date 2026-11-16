# Chuyển từ project gốc

## Những lỗi quan sát được trong mã được gửi

1. `train.py`, `train_embedding.py`, `train_triplet.py`, `evaluate.py`, `evaluate_embedding.py`
   gọi `FaceCNN(num_classes=...)` trong khi constructor gốc chỉ nhận `embedding_dim`.
2. `evaluate.py` truyền chuỗi thư mục vào Dataset yêu cầu danh sách `(path,label)` và
   lấy argmax embedding 128 chiều như nhãn lớp. Script này còn ghi checkpoint trong luồng test.
3. `triplet_dataset.py` có ký tự `e` đơn ở cuối file gây lỗi lúc import.
4. Các script dùng mẫu giả/đường dẫn cứng, lớp `minh` không khớp `dung,duy,hung`.
5. `preprocess.py` chuẩn hóa [-1,1], còn train ArcFace/embedding dùng [0,1]. Các pipeline có thể lệch nhau.
6. Luồng đã sửa trong `train_arcface.py` tốt hơn các script thử nghiệm nhưng mới hỗ trợ ba phiên.
   Bản mới mở rộng sáu phiên và gom metadata + preprocessing + prototype vào một bundle.

## Những file được bỏ khỏi bản phân phối mới

| File gốc | Lý do / nơi thay thế |
| --- | --- |
| train.py, train_embedding.py, train_triplet.py | Các thử nghiệm chưa chạy được/đường dẫn giả; dùng `faceid train` |
| pair_dataset.py, triplet_dataset.py, loss.py | Không dùng trong pipeline ArcFace được chọn; giảm nhiều nguồn chân lý |
| evaluate.py, evaluate_embedding.py | Luồng đánh giá sai hoặc hard-code; dùng `faceid evaluate` |
| train_arcface.py | Gom train/prototype/early stop và metadata vào `faceid/engine.py` |
| embedding.py, recognize.py | Gom inference và ngưỡng ràng buộc checkpoint vào package |
| collec_data.py | Giao diện camera trình duyệt thay vòng lặp desktop và tên file sai chính tả |
| dataset.py, preprocess.py, model.py, arcface.py ở thư mục gốc | Tổ chức thành các module rõ trách nhiệm trong `faceid/` |

Hai trọng số `.pth` và `class_names.txt` gốc vẫn được giữ nguyên trong `weights/legacy/`.
Các tệp tải lên không bị xóa; việc dọn dẹp áp dụng cho bản project được tái cấu trúc.

## Di chuyển dữ liệu

Cấu trúc đích là `data/faces/{dung,duy,hung}/phien_01..phien_06/`.
`python -m faceid init` chỉ tạo thư mục trống và bundle cũ, không di chuyển ảnh.
Nếu thư mục cũ tên `session_01`, hãy xác định buổi chụp rồi đổi tên rõ ràng;
không sao chép một buổi chụp thành nhiều phiên. Với ba phiên cũ, cần thu thập thêm
ba phiên độc lập để có đủ sáu. Giữ ảnh mỗi phiên trong đúng một tập.

Nếu ảnh cũ đã crop mặt 160×160, kiểm tra trực quan và audit. Detector có thể không nhận ra
một crop quá sát. Có thể đặt `crop_face=false` cho toàn bộ dataset crop sẵn khi chủ động
chuẩn hóa tất cả dữ liệu theo kiểu đó; nếu ảnh crop sát như collector cũ, đặt thêm
`face_margin=0.0` để ảnh query cũng được crop sát (không thêm lề chỉ ở inference).
Giữ `input_mode="detect_faces"` để ảnh raw ở giao diện
vẫn được phát hiện/crop trước khi suy luận. Checkpoint sẽ lưu cả hai lựa chọn. Nếu chủ động
train phân loại cả khung ảnh với `crop_face=false`, cần đặt `input_mode="raw_image"` để
query cũng được xử lý cả ảnh. Không trộn raw/crop mà không kiểm tra.

## Tương thích trọng số

`FaceCNN` mới giữ chính xác tên/shape của state_dict gốc. `import-legacy` dùng head.weight
đã chuẩn hóa làm tâm lớp, giữ resize112 và normalization none, gắn nhãn legacy chưa đánh giá.
Tâm head không phải prototype trung bình train; không được coi là bundle đã được nghiệm thu.
Không chuyển trọng số FaceCNN sang ResNet18. ResNet18 phải được train riêng; pretrained ImageNet
không đồng nghĩa pretrained nhận diện mặt. Cần so sánh bằng validation của project.

Pipeline mới thay đổi crop/resize/chuẩn hóa nên muốn đánh giá chất lượng mới phải train lại.
Không có dữ liệu ảnh trong đầu vào hiện tại để xác nhận mức cải thiện accuracy.
