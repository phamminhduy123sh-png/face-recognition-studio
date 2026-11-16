CHUAN BI UNKNOWN_VAL VA UNKNOWN_TEST

Muc dich
- Tao 20 anh nguoi la trong data/unknown_val.
- Tao 20 anh nguoi la trong data/unknown_test.
- Hai tap dung 40 danh tinh khac nhau (khong chia anh cung mot nguoi sang ca hai tap).
- Script kiem tra anh bang FaceDetector hien tai cua project truoc khi sao chep.

Cach chay
1. Giai nen file ZIP nay.
2. Copy prepare_unknown_faces.py vao thu muc goc BEN TRONG cua project:
   face-recognition-studio-main/face-recognition-studio-main/
   (thu muc co faceid/, config.json va data/).
3. Mo Terminal VS Code tai dung thu muc do.
4. Chay:
   python prepare_unknown_faces.py
5. Script se tai LFW funneled (~200 MB), giai nen vao data/external_lfw,
   sau do tao:
   data/unknown_val/unknown_001.jpg ... unknown_020.jpg
   data/unknown_test/unknown_001.jpg ... unknown_020.jpg

Luu y quan trong
- Can ket noi Internet tren may cua ban khi chay lan dau.
- Script khong ghi de anh da co trong unknown_val/unknown_test.
- Danh sach danh tinh va hash nguon duoc luu trong data/unknown_faces_manifest.csv.
- LFW la bo du lieu benchmark gom anh nguoi noi tieng thu thap tu web. Hay doc dieu khoan
  cua bo du lieu goc neu ban dinh cong bo/phan phoi lai anh; nen trich dan bai bao LFW.
- 20 anh/tap chi du de chay thu. Uoc luong FAR voi 20 anh rat tho (moi mot lan chap nhan
  sai tuong ung 5%); nen tang so nguoi la len 100-200 neu muon danh gia chac hon.
- unknown_val dung de calibrate/chon nguong. unknown_test chi dung de danh gia cuoi cung
  sau khi da chot model va nguong; khong duoc dung unknown_test de chinh nguong.
