# Báo cáo bài thực hành Ngày 1 \- Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.14.1+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên, không đổi)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| :---- | :---- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1\. Phân loại ảnh

Ảnh `traffic` (COCO 210273, 640 × 428), dòng `rank = 1`.

1. **Record hạng 1:** `class_id`\=468 ; `class_name`\=cab ; `rank`\=1 ; `score`\=0.510915 ; `taxonomy_name`\=ImageNet-1K (Top-5 còn lại: minibus 0.164, police\_van 0.086, recreational\_vehicle 0.054, streetcar 0.048.)  
2. Record nói về **cả bức ảnh**. Record không có toạ độ, box hay polygon nào- chỉ có một lớp \+ một score cho toàn ảnh, và top-5 là năm lớp xếp hạng cho cùng một ảnh. Trong khi đó ảnh traffic thực tế có nhiều xe buýt, ô tô, người (detection tìm ra bus 0.91, car 0.86…), nhưng classification chỉ trả một nhãn duy nhất "cab".  
3. Danh sách lớp **đi kèm taxonomy/checkpoint**: `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ có thể chọn trong 1000 lớp đó. Model không tự nghĩ ra lớp mới- vì vậy mới có "cab", "police\_van" (lớp ImageNet) chứ không có "bus" theo nghĩa COCO.  
4. Vì **cùng một tên có thể khác nghĩa/khác ID giữa các taxonomy** (vd. "car" trong COCO-80 có id 2, ImageNet không có lớp "car" mà tách thành cab, sports\_car…). `class_id` giúp máy đọc chính xác, `class_name` giúp người đọc hiểu, `taxonomy_name` cho biết ID đó tra theo bảng nào. Thiếu một trong ba thì khi gộp dữ liệu hoặc đổi model sẽ bị lệch nhãn.  
5. Guideline phải nói rõ **tiêu chí chọn lớp chính**, ví dụ: chọn chủ thể chiếm diện tích lớn nhất / ở tiền cảnh; hoặc chọn theo "cảnh" (street scene) thay vì từng vật; quy định thứ tự ưu tiên khi hai lớp ngang nhau; và khi nào được gắn "không xác định / escalate" thay vì đoán.  
6. Score chỉ là **độ tin của model**, không phải sự thật. Ở ảnh traffic, top-1 "cab" chỉ có 0.51 và thực tế phương tiện nổi bật là xe buýt \- model có thể tự tin mà vẫn sai, hoặc đúng mà score thấp. Ground truth phải do người gán theo guideline rồi được reviewer kiểm tra; nếu lấy prediction làm nhãn, lỗi của model sẽ bị đưa ngược vào dữ liệu huấn luyện.

## 2\. Phát hiện vật thể

Ảnh `kitchen` (COCO 397133). Record chọn: người đứng bên phải ảnh (score cao nhất trên PNG).

1. **Record:** `class_name`\=person ; `score`\=0.912625 ; `bbox_xyxy`\=\[385.33, 69.24, 498.92, 348.92\] ; `bbox_width`\=113.58 ; `bbox_height`\=279.68  
2. **Giải thích bằng lời:** gốc (0,0) ở góc trên-trái ảnh. Góc trên-trái của hộp là điểm (x\_min, y\_min)- ngang mép trái của người, ngang đỉnh đầu; góc dưới-phải là (x\_max, y\_max)- mép phải người, ngang bàn chân. Hộp rộng `bbox_width = x_max − x_min` pixel, cao `bbox_height = y_max − y_min` pixel. Hộp đứng dọc (cao \> rộng) đúng với dáng người đứng.  
3. **So ngưỡng (output ô so sánh):**  
   - `0.20` → **17** prediction  
   - `0.35` → 11 prediction (ngưỡng dùng cho JSON)  
   - `0.60` → **6** prediction  
4. Khi hạ ngưỡng, reviewer phải: (a) loại các hộp sai lớp / false positive (vd. thêm nhiều "bowl", "cup" score thấp trên mặt bàn lộn xộn); (b) gộp hộp trùng lặp cho cùng một vật; (c) kiểm tra hộp lệch biên; (d) đối chiếu với vật bị bỏ sót. Khối lượng kiểm tra tăng gần gấp 3 (6 → 17 hộp), nên cần quy trình review chặt hơn.  
5. **Quy tắc box:** *Vẽ hộp chữ nhật nhỏ nhất ôm trọn phần **nhìn thấy được** của vật thể (sát mép, không chừa nền); mỗi vật thể một hộp riêng kể cả cùng lớp; vật bị che vẫn gán nếu còn thấy ≥ \~20 % và nhận ra được lớp; không vẽ hộp cho phần bị che theo phỏng đoán.*  
6. **Vật bị che/cắt mép:** cái bàn gỗ ở tiền cảnh bị **cắt ở mép dưới-trái ảnh** và bị bát/đĩa đè lên; ngoài ra lò ("oven") bên trái bị cái bàn che một phần. Đây là tình huống **lặp lại ở rất nhiều ảnh** (vật cắt mép, bị che) nên tôi **viết thành quy tắc guideline** (như quy tắc ở câu 5: chỉ vẽ phần nhìn thấy, ghi cờ `truncated`/`occluded` nếu schema có). Chỉ escalate khi không chắc **lớp** của vật (vd. đồ vật bị che quá nhiều không nhận ra là bowl hay pot).

## 3\. Phân đoạn theo từng đối tượng

Ảnh `kitchen`. Record chọn: người đứng bên phải (person 0.90 trên PNG segmentation).

1. **Record:** `instance_id`\=kitchen-001 ; `class_name`\=person ; `score`\=0.899318 ; `polygon_point_count`\=348 ; 3–5 điểm đầu `polygon_xy`\=\[\[446, 70\], \[445, 71\], \[444, 71\], \[443, 72\], \[442, 72\]\]  
2. Box chỉ cho biết vật nằm trong một hình chữ nhật, nên chứa cả nền (tường, bàn) quanh người. Polygon **bám đúng đường viền**: cho thấy dáng người (vai, tay, hai chân tách nhau, khoảng trống giữa hai chân), từ đó tính được diện tích thật, hình dạng, và tách được người khỏi vật đứng sát cạnh.  
3. `instance_id` chỉ là **mã đánh số từng object trong output của lab này** (kitchen-001, kitchen-002…) để phân biệt hai vật cùng lớp (vd. hai người: 0.90 và 0.43). Nó **không phải class ID** vì class ID cho biết *loại* vật (person \= 0 trong COCO), còn instance\_id cho biết *cái nào*. Nó **không phải tracking ID** vì không có ý nghĩa xuyên ảnh/xuyên frame: kitchen-001 ở lần chạy này không đảm bảo là cùng vật ở ảnh khác hay lần chạy khác.  
4. **Quy tắc vẽ biên mask:** *Đặt điểm polygon bám sát mép ngoài của phần nhìn thấy được, sai lệch tối đa vài pixel; không bao phần nền hay vật khác; mỗi instance một polygon riêng; ở chỗ hai vật chạm nhau, vật nằm phía trước giữ phần biên chung, vật bị che chỉ vẽ phần lộ ra; không vẽ xuyên qua vật che.*  
5. **Chỗ khó:** trên PNG segmentation, mask **"dining table" (màu xanh) tràn qua cả bát, đĩa, đồ vật trên mặt bàn** và chạm vào mask "bowl"/"person 0.43" ở góc trái; ngoài ra có một "person 0.43" ở bên trái mà khó xác định có thật sự là người hay không. Phần **biên chồng giữa bàn và đồ trên bàn** là lỗi lặp lại → viết vào **guideline** (vật phía trước giữ biên, bàn chỉ gồm phần mặt bàn lộ ra). Riêng "person 0.43" không rõ là người hay vật → **escalate** cho Lab Coach, không tự đoán lớp.

## 4\. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| :---- | :---- | :---- | :---- | :---- |
| Phân loại ảnh | **1 nhãn / 1 ảnh**: `class_id` \+ `class_name` \+ `taxonomy_name` | Ảnh traffic có nhiều xe buýt, ô tô, người nhưng top-1 là **cab (0.51)**\- chủ thể không rõ, taxonomy ImageNet không có "bus/street scene" khớp cảnh COCO | Chọn đúng một lớp theo tiêu chí "chủ thể chính" trong guideline; escalate nếu không có lớp phù hợp | Lớp có nằm trong taxonomy không, có đúng tiêu chí chủ thể chính không, tỉ lệ đồng thuận giữa annotator |
| Phát hiện vật thể | **1 hộp / 1 vật**: lớp \+ `bbox_xyxy` (pixel, gốc trên-trái) | Số hộp đổi mạnh theo ngưỡng (**17 / 11 / 6**); nhiều bowl/cup chồng nhau score thấp (0.38–0.50) trên bàn; bàn bị cắt mép ảnh | Vẽ hộp sát phần nhìn thấy, đủ mọi vật trong danh sách lớp, mỗi vật một hộp | Hộp thiếu (bỏ sót), hộp thừa/trùng, hộp lỏng/lệch biên, sai lớp; kiểm tra vật nhỏ & bị che |
| Instance segmentation | **1 polygon / 1 instance**: `instance_id` \+ lớp \+ `polygon_xy` | Mask **dining table tràn qua bát/đĩa**; "person 0.43" không rõ là người; biên chồng giữa vật chạm nhau | Vẽ polygon bám biên phần nhìn thấy, tách riêng từng instance, xử lý biên chung theo quy tắc trước/sau | Biên có bám sát không, mask có chồng/tràn sang vật khác không, mỗi instance có ID riêng, có instance nào bị gộp |

## 5\. An toàn dữ liệu

1. **Quy tắc:** Chỉ dùng ba ảnh COCO công khai có giấy phép CC BY 2.0 do notebook tải và kiểm SHA-256, không tải ảnh cá nhân/dữ liệu nội bộ, và không ghi họ tên, MSSV hay thông tin cá nhân nào vào báo cáo/output.  
2. Nếu gặp ảnh/dữ liệu ngoài phạm vi (sai checksum, ảnh lạ, có thông tin cá nhân), tôi **dừng ngay, không xử lý tiếp và báo cho Lab Coach / giảng viên (mentor)**.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md`   
- [x] `day1_lab_outputs/classification_predictions.json`  
- [x] `day1_lab_outputs/detection_predictions.json`  
- [x] `day1_lab_outputs/segmentation_predictions.json`  
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`  
- [x] `day1_lab_outputs/visuals/classification_top5.png`  
- [x] `day1_lab_outputs/visuals/detection_predictions.png`  
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`  
- [x] Ô validation cuối notebook báo `PASS`  
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output  
- [x] Đã nộp **link repository** trên LMS
