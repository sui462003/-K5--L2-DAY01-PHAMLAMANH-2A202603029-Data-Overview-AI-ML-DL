# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:2026-11-06**

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:3.13.18/2.11.0+cpu/8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \*\*Cách dùng mẫu này:\*\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \*\*upload\*\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\_lab\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \*\*link repo\*\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\_lab\_outputs/classification\_predictions.json` và `visuals/classification\_top5.png`.  
Chỉ dùng ảnh `sample\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\_id`=468 ; `class\_name`=cab; `rank`=1 ; `score`=0.510915 ; `taxonomy\_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Record này nói về cả bức ảnh. Vì sao bạn nghĩ vậy?Đây là tác vụ Phân loại ảnh (Image Classification, thể hiện qua trường "task": "image\_classification"); mô hình chỉ tính toán phân bố xác suất cho toàn bộ khung hình chứ hoàn toàn không có thông tin tọa độ hay vị trí bounding box (bbox) của từng vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Danh sách tên lớp đi kèm taxonomy/checkpoint (cụ thể là bộ nhãn ImageNet-1K gồm 1.000 lớp cố định của checkpoint yolo11n-cls.pt), mô hình không tự nghĩ ra.
4. Vì sao cần ghi cả `class\_id`, `class\_name` và `taxonomy\_name`, không chỉ ghi mỗi tên lớp?Vì mỗi taxonomy (như ImageNet, COCO, OpenImages) có bộ mã hóa và danh sách lớp khác nhau; cùng một tên lớp có thể mang ID khác nhau tùy bộ dữ liệu. Việc ghi đủ cả 3 trường giúp xác định chính xác nguồn gốc dữ liệu (provenance), tránh nhầm lẫn khi ánh xạ nhãn và đảm bảo khả năng tái lập kết quả.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần đưa ra tiêu chí ưu tiên rõ ràng: ví dụ chọn vật thể/chủ thể chiếm diện tích khung hình lớn nhất ở trung tâm, hoặc ưu tiên chọn theo thứ bậc nghiệp vụ được định nghĩa trước (ví dụ: cảnh giao thông thì ưu tiên phương tiện nổi bật nhất).
6. Vì sao không được lấy `score` của model làm ground truth? Vì score chỉ là độ tin cậy mang tính thống kê của riêng mô hình, mô hình hoàn toàn có thể đoán sai hoặc tự tin quá mức (overconfident). Ground Truth bắt buộc phải là sự thật khách quan do con người kiểm chứng và gắn nhãn theo quy tắc.

## 2\. Phát hiện vật thể

**Mở file:** `day1\_lab\_outputs/detection\_predictions.json` và `visuals/detection\_predictions.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\_name`=person ; `score`=0.912625 ; `bbox\_xyxy`=\[385.33, 69.24, 498.92, 348.92] ; `bbox\_width`=113.58 ; `bbox\_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Hộp giới hạn có góc trên-trái nằm tại tọa độ pixel. Góc dưới-phải nằm tại tọa độ pixel (498.92, 348.92) so với gốc (0, 0) ở góc trên cùng bên trái bức ảnh.Kích thước của hộp chữ nhật có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ngưỡng 0.60: Có 6 vật thể được phát hiện (gồm 2 person, 2 bowl, 2 oven có điểm số $\\ge 0.60$).Ngưỡng 0.20: Bắt được nhiều vật thể hơn (khoảng 15 – 18 vật thể, giữ lại thêm nhiều chén/bát, ly cốc nhỏ và vật dụng phụ trong bếp).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Người reviewer phải tốn nhiều thời gian và công sức để soát lọc, loại bỏ các hộp phát hiện sai (False Positives - dương tính giả) và xử lý các hộp bị trùng lặp, vì ngưỡng thấp sẽ giữ lại nhiều dự đoán có độ tin cậy kém hoặc nhiễu.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Bounding box phải bao trọn toàn bộ phần nhìn thấy được của vật thể, 4 cạnh hộp phải tiếp xúc khít với các điểm cực biên ngoài cùng (tight bounding box); không được để khoảng trống thừa của nền quá 2–3 pixel và không được cắt lẹm vào ranh giới vật thể.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Chỉ ra vật thể: Chiếc bát hoặc chén bị che khuất một phần trên kệ bếp (hoặc người/đồ đạc bị cắt lẹm ở rìa mép trái màn hình). Hướng xử lý:

\-Nếu vật thể bị che khuất một phần nhưng mắt thường vẫn nhận diện rõ ràng bản chất đối tượng: Đưa vào Guideline quy định cụ thể (ví dụ: "Chỉ đóng box bao quanh phần diện tích nhìn thấy được của vật thể, không cố đoán phần bị che khuất").

\-Nếu vật bị cắt cụt sát mép ảnh hoặc che khuất quá 80% khiến không thể xác định chính xác ranh giới hay định danh lớp: Cần đưa lên Escalation để hỏi Lab Coach thống nhất tiêu chuẩn loại bỏ hay giữ lại.

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\_lab\_outputs/segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\_id`=kitchen-001; `class\_name`=person; `score`=0.899318 ; số điểm polygon=348`polygon\_point\_count`=348 ; 3–5 điểm đầu của `polygon\_xy`=\[\[446.0, 70.0], \[445.0, 71.0], \[444.0, 71.0], \[443.0, 72.0], \[442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Bounding box chỉ là một khung chữ nhật bao ngoài, trong khi polygon cung cấp chuỗi tọa độ chính xác bám theo đường viền thực tế (contour/mask) của vật thể. Nhờ đó, ta tách được ranh giới pixel cụ thể của chủ thể và loại bỏ hoàn toàn các vùng trống/nền không thuộc về vật thể mà box vô tình ôm vào.
3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?Mục đích: Dùng để định danh và phân biệt riêng lẻ từng đối tượng cá thể (instance) độc lập xuất hiện trong cùng một bức ảnh.
Không phải class ID: Vì các vật thể cùng một lớp (ví dụ trong dữ liệu có kitchen-002, kitchen-003, kitchen-006 đều là bowl) vẫn có instance\_id khác nhau hoàn toàn.
Không phải tracking ID: Vì nó chỉ là chỉ số thứ tự tạm thời trong khuôn khổ một bức ảnh tĩnh cụ thể, không mang ý nghĩa theo dõi hay liên kết hành trình của cùng một đối tượng xuyên suốt qua chuỗi khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. Đường biên đa giác (polygon) phải bám khít mép nhìn thấy ngoài cùng của vật thể; các đường cong phức tạp cần chấm nhiều điểm liên tiếp phân bố đều để mask giữ được độ mềm mại, dung sai không vượt quá 2 pixel và không được cắt lẹm vào phần thực thể hoặc lấn ra ngoài phông nền.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Vị trí quan sát: Vùng tiếp giáp giữa chân lò nướng (oven - kitchen-007) và mặt bàn bếp/bàn ăn (dining table - kitchen-005), hoặc phần bóng đổ dưới các chiếc bát (bowl) trên kệ bếp. Hướng xử lý:



Với trường hợp hai vật chạm nhau có ranh giới phân tách vật lý rõ ràng (như bát đặt trên bàn, nồi/lò trên kệ): Sử dụng Guideline quy định nguyên tắc (ví dụ: "Khi hai vật thể chạm nhau, vẽ hai polygon độc lập tiếp xúc sát cạnh; ưu tiên vật thể ở phía trước đóng trọn đường nét, vật bị che vẽ tới mép giao cắt").



Với trường hợp vùng quá tối, bóng đổ mờ mịt hoặc chi tiết bị cháy sáng/mờ nét không thể phân biệt ranh giới pixel thuộc về vật nào: Cần đưa lên Escalation để hỏi Lab Coach thống nhất tiêu chuẩn gán nhãn chung, tránh việc người gán nhãn tự phỏng đoán chủ quan.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Một nhãn duy nhất gồm class\_id và class\_name từ taxonomy cho toàn bộ bức ảnh (không có tọa độ vị trí).|Ảnh traffic có rất nhiều đối tượng (xe buýt, ô tô, người), mô hình đoán lớp cab chỉ đạt score 0.510915 (top 2 là minibus 0.164284); khó chọn đúng 1 nhãn bao quát toàn cảnh.|Xem ảnh toàn cảnh và chọn đúng một lớp chủ đề/đối tượng chiếm ưu thế chính theo tiêu chuẩn trong guideline.|Soát lại ảnh và guideline xem lớp được chọn có đúng quy tắc ưu tiên không, đảm bảo không nhầm lẫn giữa các lớp tương tự nhau.|
|Phát hiện vật thể|Danh sách các hộp chữ nhật gồm class\_id, class\_name và tọa độ 4 góc pixel bbox\_xyxy bao quanh từng vật thể.|Tại ngưỡng 0.60 chỉ bắt được 6 đối tượng lớn, bỏ sót nhiều bát/cốc nhỏ trên kệ bếp (kitchen); một số box ôm cả phần bị che khuất.|Vẽ bounding box tiếp xúc khít 4 cạnh ngoài của từng vật thể nhìn thấy được và gán đúng lớp.|Soát lỗi sót vật thể (False Negative), kiểm tra box có bị lỏng (thừa nền) hoặc cắt lẹm vào vật thể hay không.|
|Instance segmentation|Tập hợp các đỉnh đa giác pixel polygon\_xy riêng biệt cho từng cá thể kèm instance\_id và nhãn lớp.|Mask của bàn ăn (dining table - kitchen-005) có tới 598 điểm bị răng cưa, lem lấn vào vùng sàn và chân lò nướng (kitchen-007).|Chấm các điểm đa giác men sát đường viền ngoài của từng thực thể riêng biệt, tách rời các vật nằm cạnh nhau.|Kiểm tra độ khít của mask viền, lỗi lem sang nền/vật thể khác, hoặc việc gộp nhầm hai cá thể cạnh nhau thành một mask.|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) Chỉ sử dụng 3 ảnh mẫu COCO công khai có bản quyền hợp lệ theo quy định của bài lab; tuyệt đối không đưa hình ảnh cá nhân, thông tin định danh cá nhân (PII), hay dữ liệu nội bộ/khách hàng vào báo cáo và mã nguồn trên repository GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Dừng ngay công việc xử lý dữ liệu và báo cáo trực tiếp cho Lab Coach / Giảng viên hướng dẫn để được hỗ trợ cô lập và xử lý trường hợp bất thường.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

