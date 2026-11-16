# Face Studio — Dũng · Duy · Hưng

Ứng dụng nhận diện khuôn mặt chạy trên máy cá nhân: chụp dữ liệu theo phiên,
kiểm tra dữ liệu, huấn luyện, xem báo cáo và nhận diện từ ảnh/webcam trình duyệt.
Giao diện và CLI dùng **cùng một pipeline**; dữ liệu ảnh không được gửi tới dịch vụ bên ngoài.

## Chạy nhanh

Yêu cầu Python 3.10–3.12, máy có webcam nếu muốn chụp ảnh. CPU chạy được;
GPU CUDA giúp huấn luyện nhanh hơn. Trên Windows có thể chạy `start.bat`.
Trên Linux/macOS chạy `sh start.sh`, hoặc thực hiện thủ công:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m faceid init
python -m faceid serve
```

Mở **http://127.0.0.1:8000**. Mặc định chỉ lắng nghe trên máy cá nhân.
Cho phép quyền camera trong trình duyệt để chụp dữ liệu/nhận diện.
Camera trên máy người dùng được đọc bằng trình duyệt, không phải camera trên server.
`127.0.0.1` luôn chỉ máy đang mở trình duyệt, không phải địa chỉ website công khai.
Phải tải/giải nén project và chạy server trên máy đó trước. `start.bat` đợi server
khởi động rồi mới mở trình duyệt; giữ nguyên cửa sổ lệnh khi dùng ứng dụng.
Nếu khởi động lỗi, cửa sổ sẽ giữ lại thông báo để kiểm tra.

`init` tạo thư mục và chuyển hai trọng số gốc đi kèm thành `artifacts/legacy.pth`.
Mô hình cũ có thể hiển thị **ứng viên**, nhưng chưa được coi là đã đánh giá hay đã
hiệu chỉnh ngưỡng. Thiếu dữ liệu ảnh thật nên bản phân phối **không tuyên bố accuracy**.
Không có bản huấn luyện giả hoặc số liệu giả được gắn nhãn thành kết quả thật.

## Dữ liệu: ba người × sáu phiên

```text
data/
  faces/
    dung/
      phien_01/ ... phien_06/
    duy/
      phien_01/ ... phien_06/
    hung/
      phien_01/ ... phien_06/
  unknown_val/   # Người ngoài 3 lớp, chỉ để chọn ngưỡng
  unknown_test/  # Người ngoài, phiên độc lập, để kiểm chứng sau cùng
```

Nhãn cố định: **dung=0, duy=1, hung=2**; tên hiển thị Dũng, Duy, Hưng.
Mỗi ảnh có một người và đúng nhãn. Mỗi phiên là một buổi chụp/video riêng, có
sự thay đổi ánh sáng, nền, khoảng cách và hướng nhìn. Nên có vài chục ảnh đa dạng
ở mỗi phiên; nhiều frame gần giống nhau không thay thế được nhiều điều kiện chụp.
Tránh chia ngẫu nhiên các frame của cùng video sang nhiều phiên.

Mặc định áp dụng cho cả ba người:

| Tập | Phiên | Mục đích |
| --- | --- | --- |
| Train | 01, 02, 03, 04 | Học trọng số và tạo prototype |
| Validation | 05 | Chọn checkpoint và hiệu chỉnh ngưỡng |
| Test | 06 | Đánh giá sau khi chốt mọi tham số |

Đây là tỷ lệ theo **phiên**, không bảo đảm tỷ lệ số ảnh khi phiên có kích thước khác nhau.
Có thể đổi `splits` trong `config.json`; mọi phiên phải thuộc đúng một tập, cả ba tập không rỗng.
Thư mục thiếu/rỗng, ảnh hỏng, ảnh sai cấp, nhiều mặt, hoặc ảnh trùng giữa các tập/người
được báo rõ. Hash byte và pixel chỉ phát hiện bản sao chính xác, không thay thế kiểm tra ảnh gần giống.

Không có ảnh gốc trong các file bạn cung cấp, vì vậy `data/` đi kèm chỉ là khung thư mục.
Ứng dụng không tự tạo sáu “phiên thực” từ một buổi chụp, và không tự di chuyển ảnh cũ.
Nếu dữ liệu cũ dùng `session_01`…`session_03`, xem [MIGRATION.md](docs/MIGRATION.md).

## Tiền xử lý thống nhất

`faceid/preprocess.py`: đọc ảnh → sửa EXIF → RGB (alpha ghép nền trắng) →
phát hiện đúng một khuôn mặt → crop thêm lề → resize giữ tỷ lệ/thêm nền trắng →
augmentation **chỉ cho train** → tensor float32 CHW → normalization.

Validation/test, prototype và suy luận không augmentation. Khi webcam đã phát hiện
từng mặt, tensor hóa nhận `already_cropped=True` để không phát hiện/crop lần hai.
Không phát hiện mặt thì mặc định báo lỗi; chỉ dùng `no_face="keep"` khi đã kiểm tra
dữ liệu crop sẵn và chủ động chấp nhận ảnh đó. Không dùng ảnh sai nhãn để cố đủ số lượng.

ResNet18 pretrained mặc định: `224×224`, chuẩn hóa ImageNet. FaceCNN có thể chọn
`none` ([0,1]) hoặc `symmetric` ([-1,1]); mọi cấu hình được lưu cùng checkpoint.
Thay tiền xử lý rồi dùng nguyên trọng số cũ không bảo đảm chất lượng; cần huấn luyện/đánh giá lại.
Haar Cascade chạy offline, nhanh nhưng có thể bỏ sót mặt nghiêng/mặt nhỏ; không có bước căn chỉnh landmark.

## Quy trình huấn luyện và kiểm chứng

```bash
# 1. Kiểm tra toàn bộ dữ liệu trước khi train
python -m faceid audit

# 2. ResNet18 ImageNet + embedding + ArcFace
python -m faceid train --epochs 40

# Hoặc fine-tune FaceCNN từ backbone cũ để so sánh trên validation
python -m faceid train --backbone facecnn --warm-start weights/legacy/arcface_backbone.pth

# 3. Cần ảnh người lạ riêng trong data/unknown_val để chọn ngưỡng
python -m faceid calibrate --unknown-dir data/unknown_val --max-far 0.05

# 4. Chỉ sau khi chốt model/ngưỡng: đánh giá test
python -m faceid evaluate --split test

# 5. Nhận diện ảnh (hoặc dùng giao diện)
python -m faceid predict path/to/photo.jpg
```

Lần train ResNet18 pretrained đầu tiên cần Internet để tải trọng số ImageNet chính thức
của torchvision (~45 MB). Chạy offline được với `--offline`, nhưng khi đó ResNet18 khởi
tạo ngẫu nhiên; không nên kỳ vọng hiệu quả transfer learning. Không tải pretrained lúc mở ứng dụng.

Huấn luyện dùng AdamW, scheduler, gradient clipping, early stopping; chọn checkpoint theo
validation, không dùng test để chọn epoch. Prototype được tính từ **ảnh train không augmentation**.
Checkpoint chứa kiến trúc, trọng số, prototype, thứ tự nhãn, pipeline, phiên và dấu vân tay dữ liệu.
Các bộ phận phải khớp nhau khi suy luận; ngưỡng được gắn với hash checkpoint và pipeline.

Không có ảnh người lạ validation thì không có ngưỡng Unknown đã kiểm chứng. Giao diện vẫn cho
xem ứng viên với nhãn “Chưa hiệu chỉnh”, không gán một ngưỡng 0.8 tùy ý rồi gọi là độ chính xác.
Cosine similarity **không phải xác suất**. FAR ≤ mục tiêu trên validation chỉ là kết quả thực nghiệm
của tập đó, không phải bảo đảm ngoài thực tế. Cần đo thêm trên người lạ test độc lập khi triển khai.

## Giao diện và đầu ra

- **Nhận diện**: tải ảnh hoặc camera, vẽ khung từng mặt, hiển thị ứng viên/Unknown và cosine.
- **Dữ liệu**: thống kê 18 phiên, chọn người/phiên, chụp và lưu ảnh gốc có đúng một khuôn mặt.
- **Huấn luyện**: chạy một tác vụ nền, theo dõi tiến độ và lỗi; không chạy nhiều job chồng nhau.
- **Báo cáo**: đọc các báo cáo có thật; trạng thái trống khi chưa huấn luyện/đánh giá.
- **Hướng dẫn**: các bước chạy và nguyên tắc tách phiên.

Chụp lưu ảnh gốc trước khi vẽ khung, không ghi đè ảnh cũ. Bảo đảm một phiên thật không bị trộn
với nhiều lần chụp khác nhau; việc chọn phiên là trách nhiệm của người thu thập.
`artifacts/` chứa các file được tạo khi sử dụng, không phải nguồn mã. `best.pth` là bundle mới;
`legacy.pth` là bản gói trọng số cũ và luôn được đánh dấu chưa nghiệm thu.
Mỗi lần train mới, checkpoint/ngưỡng/báo cáo của lần trước được chuyển vào
`artifacts/archive/` để không lẫn kết quả giữa hai mô hình.

## Cấu trúc mã nguồn

```text
faceid/
  config.py         # Cấu hình JSON, nhãn và đường dẫn
  preprocess.py     # Detector + crop + augmentation + tensor/normalize
  data.py           # Quét/audit/chia phiên/Dataset
  models.py         # FaceCNN tương thích trọng số cũ + ResNet18 + ArcFace
  engine.py         # Train/prototype/evaluate/import legacy
  inference.py      # Nạp bundle, nhận diện, kiểm tra ngưỡng
  calibration.py    # Hiệu chỉnh Unknown chỉ trên validation
  cli.py            # Một CLI cho mọi thao tác
  web/              # API FastAPI + giao diện HTML/CSS/JS nội bộ
weights/legacy/     # Hai trọng số và class_names gốc bạn cung cấp
tests/              # Kiểm tra hành vi và luồng tích hợp
docs/               # Chuyển đổi dự án và báo cáo kiểm chứng
```

Xem [MIGRATION.md](docs/MIGRATION.md) để biết những script đã loại khỏi bản phân phối mới.
Không xóa bản gốc người dùng tải lên; project mới được dựng riêng.

## Kiểm tra tự động

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Kiểm tra dùng ảnh tổng hợp cho tính đúng đắn phần mềm, không dùng chúng để kết luận nhận diện mặt tốt.
Xem [VALIDATION.md](docs/VALIDATION.md) để phân biệt kiểm thử đã chạy với đánh giá còn cần ảnh thật.

## Xử lý lỗi thường gặp

- **Audit báo phiên trống**: điền ảnh cho đủ 18 thư mục; `init` chỉ tạo thư mục.
- **No face / nhiều mặt**: chụp rõ một khuôn mặt; ảnh crop cũ cần kiểm tra trước khi đổi chính sách no_face.
- **Ngưỡng thuộc model khác**: hiệu chỉnh lại sau khi train; không tái dùng ngưỡng với checkpoint khác.
- **Không tải được pretrained**: kiểm tra Internet hoặc chủ động chạy FaceCNN/warm start; không có fallback âm thầm.
- **Camera bị chặn**: mở trên localhost, cấp quyền camera; đóng ứng dụng khác đang chiếm camera.
- **CPU train chậm**: giảm batch/epochs để thử chức năng; dùng PyTorch CUDA theo hướng dẫn chính thức khi có GPU.
- **Train accuracy cao nhưng val thấp**: thêm phiên đa dạng, kiểm tra nhãn/crop và rò rỉ, giảm overfitting;
  không chuyển ảnh test vào train để làm số liệu đẹp hơn.

Ứng dụng nhận diện ba người đã đăng ký; đây không phải hệ thống chống giả mạo/liveness hoặc chấm công đã nghiệm thu.
