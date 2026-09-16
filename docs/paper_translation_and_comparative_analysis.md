# BẢN DỊCH CHI TIẾT VÀ PHÂN TÍCH ĐỐI SÁNH 2 BÀI BÁO QUỐC TẾ
## Ứng dụng vào đề tài: "Sinh báo cáo MRI Cột sống thắt lưng theo mức có hiệu chuẩn độ bất định"

---

# PHẦN 1: BÀI BÁO 1 — C2M-DoT (Medical Image Analysis, 2023)

- **Tiêu đề gốc:** *C2M-DoT: Cross-modal consistent multi-view medical report generation with domain transfer network*
- **Tác giả:** Ruizhi Wang, Xiangtao Wang, Jie Zhou, Thomas Lukasiewicz, Zhenghua Xu.
- **Đơn vị nghiên cứu:** Đại học Công nghệ Hà Bắc (Trung Quốc), Đại học Oxford (Vương quốc Anh), TU Wien (Áo).
- **Tạp chí công bố:** *Medical Image Analysis (MedIA)*, 2023 (Q1, IF ~10.9, Top 1 tạp chí AI y tế thế giới).

---

### 1.1. Tóm tắt (Abstract)
Trong các kịch bản lâm sàng, nhiều hình ảnh y tế với các góc chụp (views) khác nhau thường được tạo ra đồng thời và các hình ảnh này có tính nhất quán ngữ nghĩa rất cao. Tuy nhiên, phần lớn các phương pháp sinh báo cáo y tế hiện nay chỉ xem xét dữ liệu đơn góc nhìn (single-view). 

Thông tin tương hỗ phong phú giữa các góc nhìn có thể giúp sinh báo cáo chính xác hơn. Dẫu vậy, sự phụ thuộc của mô hình đa góc nhìn vào dữ liệu đa góc nhìn ở giai đoạn suy luận (inference) làm hạn chế nghiêm trọng khả năng ứng dụng trong thực tế lâm sàng (nơi có thể thiếu một số góc chụp). Ngoài ra, các phương pháp tối ưu hóa cấp độ từ (word-level cross-entropy) chỉ dựa trên xác suất rời rạc, bỏ qua sự đồng nhất ngữ nghĩa giữa báo cáo và ảnh y tế.

Để giải quyết vấn đề này, bài báo đề xuất **C2M-DoT**:
1. **Học tương phản đa góc nhìn dựa trên ngữ nghĩa (Semantic-based Multi-view Contrastive Learning - S-MCL / MvCo)** để tận dụng thông tin chéo giữa các góc chụp nhằm biểu diễn tổn thương.
2. **Mạng chuyển dịch miền (Domain Transfer Network - DoT)** đảm bảo mô hình huấn luyện đa góc nhìn vẫn duy trì hiệu năng suy luận xuất sắc khi chỉ có đầu vào là đơn góc nhìn.
3. **Hàm mất mát nhất quán xuyên phương thức (Cross-Modal Consistency Loss - CMCL)** căn chỉnh ngữ nghĩa giữa báo cáo văn bản và đặc trưng hình ảnh y tế thông qua mô hình thị giác - ngôn ngữ y tế (PubMedCLIP).

Thực nghiệm trên 2 tập dữ liệu benchmark lớn (IU X-Ray và MIMIC-CXR) chứng minh C2M-DoT vượt trội hơn tất cả các phương pháp SOTA trên toàn bộ các chỉ số đánh giá.

---

### 1.2. Kiến trúc và Phương pháp Luận Chi Tiết (Methodology)

#### A. Multi-View Contrastive Learning (MvCo)
- **Vấn đề:** Các kỹ thuật contrastive learning trước đây chủ yếu áp dụng ở tầng tiền huấn luyện (pre-training) cho bộ trích xuất đặc trưng thị giác, nhưng đặc trưng cấp thấp thường khó khái quát hóa tốt cho nhiệm vụ sinh báo cáo (downstream task).
- **Giải pháp:** C2M-DoT đưa Contrastive Learning trực tiếp vào quy trình sinh báo cáo end-to-end ở **tầng giải mã ngữ nghĩa (decoded semantic embeddings)**.
  - Ảnh mặt trước ($I_{frontal}$) và mặt nghiêng ($I_{lateral}$) được trích xuất đặc trưng không gian bởi ResNet-101 thành $f_{frontal}$ và $f_{lateral}$.
  - Chiếu sang không gian biểu diễn phân biệt:
    $$F_{frontal} = \phi_f(f_{frontal}), \quad F_{lateral} = \phi_l(f_{lateral})$$
  - Sau khi đi qua mạng giải mã M-Linear Transformer, thu được vector ngữ nghĩa ngữ cảnh $c$ và vector ẩn $h$. Kết hợp lại và chiếu vào không gian ẩn:
    $$x_f = \psi([c_f, h_f]), \quad x_l = \psi([c_l, h_l])$$
  - Hàm mất mát tương phản đa góc nhìn InfoNCE ($L_{MvCo}$):
    $$L_{MvCo} = -\log \frac{\exp(\text{sim}(x_l, x_f)/\tau_c)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq l]} \exp(\text{sim}(x_l, x_k)/\tau_c)}$$
    *Ý nghĩa:* Tối đa hóa sự tương đồng ngữ nghĩa giữa góc chụp thẳng và góc chụp nghiêng của **cùng một bệnh nhân**, đồng thời phân tách sự tương đồng giữa các bệnh nhân khác nhau.

#### B. Domain Transfer Network (DoT) — Xử lý bài toán thiếu góc chụp khi suy luận
- **Vấn đề thực tế (Domain Shift):** Khi train dùng cả 2 góc chụp ($F_{frontal} + F_{lateral}$), nhưng khi đi khám thực tế, bệnh nhân chỉ có 1 góc chụp (ví dụ chỉ chụp thẳng), mô hình đa góc nhìn thông thường sẽ sụp đổ hiệu năng vì lệch phân phối đầu vào.
- **Không gian hành động (Action Space):**
  $$a_i = \begin{cases} F_{frontal}, & i=0 \\ F_{lateral}, & i=1 \\ F_{fusion} = F_{frontal} + F_{lateral}, & i=2 \end{cases}$$
- **Cơ chế tái tham số hóa Gumbel-Softmax:**
  - Mô hình dùng một mạng chấm điểm confidence:
    $$P = \text{softmax}(W_c [F_{frontal}, F_{lateral}])$$
  - Để có thể lan truyền ngược vi phân (differentiable backpropagation) qua quá trình lựa chọn rời rạc, mô hình áp dụng Gumbel-Softmax trick:
    $$V(a) = \frac{\exp((\log P_i(a) + g_i)/\tau_s)}{\sum_j \exp((\log P_j(a) + g_j)/\tau_s)}$$
  - *Ý nghĩa:* Mô hình tự học cách thích nghi linh hoạt: lúc thì nhìn ảnh thẳng, lúc thì nhìn ảnh nghiêng, lúc thì nhìn kết hợp cả hai. Nhờ đó khi test, nếu chỉ đưa vào 1 ảnh, mô hình vẫn suy luận chính xác mà không bị suy giảm hiệu năng!

#### C. Cross-Modal Consistency Loss ($L_{CMC}$)
- **Vấn đề:** Hàm Cross-Entropy ($L_{CE}$) truyền thống chỉ so sánh từng token độc lập, hoàn toàn bỏ qua ngữ nghĩa lâm sàng toàn cục của ảnh và văn bản.
- **Giải pháp:** Sử dụng PubMedCLIP (ViT-B/32 pre-trained trên ROCO) để trích xuất embedding thị giác ($v_f, v_l$) và embedding văn bản ($t_{pred}, t_{true}$).
- Tính ma trận phân phối xác suất tương đồng cosine hai chiều giữa ảnh và text (Image-to-Text và Text-to-Image).
- Dùng khoảng cách Kullback-Leibler (KL Divergence) để ép phân phối tương đồng giữa **báo cáo do AI sinh ra với ảnh** phải tiệm cận phân phối tương đồng giữa **báo cáo thật của bác sĩ với ảnh**:
  $$L_{CMC}^F = \frac{1}{2} \mathbb{E}\left[ \text{KL}(S_{v2t}^{F, pred} \parallel S_{v2t}^{F, true}) + \text{KL}(S_{t2v}^{F, pred} \parallel S_{t2v}^{F, true}) \right]$$

---

### 1.3. Kết quả Thực nghiệm Chính của C2M-DoT
- **IU X-Ray:** BLEU-4 đạt **0.198**, ROUGE-L đạt **0.402**, METEOR đạt **0.201** (vượt xa R2Gen và HReMRG-MR).
- **MIMIC-CXR:** BLEU-4 đạt **0.126**, ROUGE-L đạt **0.298**.
- **Ablation Study:** Chứng minh rằng nếu bỏ DoT, khi test trên ảnh đơn góc nhìn, BLEU-4 tụt giảm hơn 28%; nếu bỏ $L_{CMC}$, các câu sinh ra lặp từ và mô tả sai lệch tổn thương.

---

### 1.4. [NOTE QUAN TRỌNG CHO BÀI BÁO CỦA BẠN]
1. **Tương đồng trực tiếp với bài toán MRI Cột sống (Sagittal vs Axial = Frontal vs Lateral):**
   - Trong MRI cột sống, lát cắt Sagittal (dọc giữa) cho thấy hình thái thân đốt, chiều cao đĩa đệm, độ cong sinh lý và ống sống.
   - Lát cắt Axial (ngang) cho thấy lỗ liên hợp hai bên, ngách bên và mức độ chèn ép rễ thần kinh.
   - Hai góc cắt này hoàn toàn tương ứng với cặp *Frontal - Lateral* trong C2M-DoT.
2. **Giải quyết bài toán "Thiếu lát cắt Axial" (Missing Slices) trong thực tế:**
   - Trong bộ dữ liệu MRI thực tế (như PSPINES hoặc RSNA 2024), rất nhiều bệnh nhân **chỉ được quét lát cắt ngang Axial ở các tầng nghi ngờ bệnh (ví dụ L4/L5, L5/S1)**, còn tầng L1/L2, L2/L3 hoàn toàn không có ảnh Axial!
   - Kỹ thuật Domain Transfer Network (DoT) hoặc cơ chế Adaptive Input Routing của C2M-DoT là cơ sở lý thuyết cực kỳ đắt giá để giải thích cách hệ thống của bạn sinh báo cáo cho các tầng thiếu ảnh Axial mà không bị lỗi hoặc suy giảm hiệu năng.
3. **Ý tưởng hàm Loss nhất quán đa phương thức ($L_{CMC}$):**
   - Trong Phase V2 của bạn, khi muốn phạt hiện tượng ảo giác (hallucination), bạn hoàn toàn có thể trích dẫn C2M-DoT để lý giải vì sao không nên chỉ dùng Cross-Entropy đơn thuần mà cần có thành phần căn chỉnh ngữ nghĩa chéo.

---
---

# PHẦN 2: BÀI BÁO 2 — ViMed-PET (NeurIPS 2025 Datasets & Benchmarks Track)

- **Tiêu đề gốc:** *Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation*
- **Tác giả:** Huu Tien Nguyen, Dac Thai Nguyen, The Minh Duc Nguyen, Trung Thanh Nguyen, Huy Hieu Pham, Johan Barthelemy, Minh Quan Tran, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Hong Son Mai, Thanh Trung Nguyen, Phi Le Nguyen.
- **Đơn vị nghiên cứu:** Viện Nghiên cứu AI4LIFE (ĐH Bách Khoa Hà Nội), VinUniversity, Bệnh viện Trung ương Quân đội 108, ĐH Y Hà Nội, NVIDIA (Mỹ), Griffith University (Úc).
- **Hội nghị công bố:** *NeurIPS 2025* (Top 1 Hội nghị Trí tuệ Nhân tạo thế giới, Track Datasets & Benchmarks).

---

### 2.1. Tóm tắt (Abstract)
Các mô hình thị giác - ngôn ngữ nền tảng (VLMs) trong y tế hiện nay hầu hết chỉ tập trung vào ảnh 2D thông thường (như X-quang) và ngôn ngữ tài nguyên cao (tiếng Anh). Các phương thức hình ảnh chức năng 3D (như PET/CT) và các ngôn ngữ ít tài nguyên (như tiếng Việt) gần như chưa được khai phá.

Nhóm tác giả công bố tập dữ liệu y tế đa phương thức tiếng Việt đầu tiên gồm **2.757 ca chụp PET/CT toàn thân 3D** từ các bệnh nhân độc lập cùng báo cáo lâm sàng hoàn chỉnh từ **Bệnh viện Trung ương Quân đội 108**. Bài báo đóng góp:
1. Tập dữ liệu ảnh PET/CT 3D - báo cáo tiếng Việt quy mô lớn đầu tiên.
2. Khung phân chia giải phẫu 3 miền cơ thể (Head-Neck, Chest, Abdomen-Pelvis) giúp tăng quy mô mẫu lên 8.271 cặp và tăng cường khả năng căn chỉnh cục bộ.
3. Pipeline huấn luyện 3 giai đoạn (Stage 1: Pretrain 3D Vision Encoder $\rightarrow$ Stage 2: Feature Alignment $\rightarrow$ Stage 3: Instruction-Tuning LoRA trên LLM).
4. Hệ thống chỉ số đánh giá lâm sàng mới (**Clinical F1-Scores**: F1-T, F1-TP, F1-TF, F1-TPF) được thẩm định bởi bác sĩ chuyên khoa.

---

### 2.2. Chi Tiết Phương Pháp và Đóng Góp Kỹ Thuật

#### A. Xử lý dữ liệu và Phân vùng Giải phẫu (Body Part Partition)
- **Vấn đề:** Thể tích PET/CT 3D toàn thân chứa hàng trăm lát cắt; nếu đưa cả khối vào VLM sẽ tràn bộ nhớ GPU và làm loãng thông tin chẩn đoán cục bộ.
- **Phân đoạn giải phẫu thích ứng:**
  - Tách mỗi ca thành 3 vùng: **Đầu - Cổ** (Head-Neck, ~20% chiều cao trên), **Ngực** (Chest, bắt đầu 15% dưới cổ), và **Bụng - Chậu** (Abdomen-Pelvis, phần còn lại).
  - Đưa vào vùng chồng lấn **20 lát cắt (20-slice overlap)** giữa các đoạn lân cận để tránh mất mát thông tin tại ranh giới giải phẫu.
  - Tăng tổng số cặp dữ liệu từ 2.757 lên **8.271 cặp ảnh-báo cáo**.

#### B. Khung Huấn luyện 3 Giai đoạn (3-Stage Training Flow)
- **Stage 1 (Fine-tuning 3D Vision Encoder):** Sử dụng **CT-ViT** (pretrained trên CT 3D) và **Cosmos Tokenizer** (bỏ attention nhân quả video để chuyển thành spatial 3D) để học biểu diễn thể tích 3D y tế.
- **Stage 2 (Concept Feature Alignment):** Đóng băng (freeze) cả Vision Encoder và LLM; chỉ huấn luyện bộ chiếu tuyến tính 3D (3D Linear Projector) để căn chỉnh không gian embedding thị giác vào không gian token của LLM.
- **Stage 3 (Instruction-Tuning with LoRA):** Đóng băng Vision Encoder, huấn luyện bộ chiếu và gắn LoRA vào Large Language Model (Mistral-7B / LLaMA-2-7B).

#### C. Thước đo Đánh giá Lâm sàng (Clinical F1 Metrics)
- Nhóm chỉ ra rằng BLEU và ROUGE không phản ánh đúng giá trị chẩn đoán y khoa (ví dụ: một báo cáo đạt BLEU cao do câu từ trau chuốt nhưng ghi sai vị trí khối u thì hoàn toàn vô dụng).
- Nhóm xây dựng tập test y tế gồm 80 bệnh nhân ung thư phổi với 398 tổn thương được dán nhãn chuẩn vàng bởi bác sĩ:
  - **F1-T:** Độ chính xác về Loại tổn thương (Lesion Type).
  - **F1-TP:** Loại tổn thương + Vị trí giải phẫu (Type + Position).
  - **F1-TF:** Loại tổn thương + Mức độ chuyển hóa phóng xạ FDG.
  - **F1-TPF:** Kết hợp đồng thời cả Loại + Vị trí + Mức độ chuyển hóa.

---

### 2.3. Kết quả Thực nghiệm Nổi Bật của ViMed-PET
1. **Các mô hình VLM y tế tiếng Anh mã nguồn mở (Zero-shot / Off-the-shelf):**
   - LLaVA-Med: BLEU-4 = **0.01**, ROUGE-L = **27.89**.
   - M3D (3D Medical): BLEU-4 = **0.04**, ROUGE-L = **23.53**.
   - RadFM: BLEU-4 = **0.06**, ROUGE-L = **28.33**.
   $\rightarrow$ *Kết luận giật mình:* Các mô hình VLM y tế tiếng Anh tốt nhất thế giới hoàn toàn tê liệt khi sinh văn bản tiếng Việt nếu không được tinh chỉnh trực tiếp!
2. **GPT-4o (Few-shot Prompting):**
   - BLEU-4 = **31.12**, ROUGE-L = **52.76**, Clinical F1-TPF = **7.87%**.
3. **Mô hình Fine-tuned trên ViMed-PET (CT-ViT + Mistral-7B LoRA):**
   - BLEU-4 đạt **58.07**, ROUGE-L đạt **72.74**, BERTScore đạt **89.98**.
   - Clinical F1-TPF đạt **22.65%** (cao gấp gần 3 lần GPT-4o!).

---

### 2.4. [NOTE QUAN TRỌNG CHO BÀI BÁO CỦA BẠN]
1. **Là tài liệu tham khảo và đối sánh nền tảng cao nhất (State-of-the-Art) tại Việt Nam:**
   - ViMed-PET vừa được công bố tại NeurIPS 2025. Đây là bài báo chuẩn mực nhất về sinh báo cáo y tế tiếng Việt đa phương thức. Trích dẫn và áp dụng phương pháp luận của ViMed-PET sẽ giúp bài báo của bạn có độ tin cậy học thuật cực kỳ vững chắc.
2. **Đối sánh Phương pháp Phân vùng: 3 Miền cơ thể vs 5 Tầng đĩa đệm (Level-wise):**
   - ViMed-PET chia PET/CT toàn thân thành 3 miền giải phẫu (Head-Neck, Chest, Abdomen-Pelvis) để giải quyết vấn đề loãng thông tin.
   - Bài báo của bạn phân chia cột sống thắt lưng thành **5 mức đĩa đệm độc lập (L1-L2 đến L5-S1)**. Bạn có thể trích dẫn trực tiếp cách tiếp cận của ViMed-PET để bảo vệ tính đúng đắn khoa học của cấu trúc Level-wise.
3. **Phương pháp đánh giá lâm sàng (Clinical F1):**
   - Thay vì chỉ báo cáo BLEU/ROUGE (vốn bị coi là thiếu chiều sâu y khoa), bạn hãy xây dựng bộ chỉ số **Clinical F1 theo độ thoái hóa đĩa đệm (Pfirrmann grade)**, **hẹp ống sống (Canal stenosis grade)** và **hẹp lỗ liên hợp (Foraminal stenosis grade)** tương tự như F1-T / F1-TP / F1-TPF của ViMed-PET.

---
---

# PHẦN 3: PHÂN TÍCH CHIẾN LƯỢC NGÔN NGỮ
## Trả lời câu hỏi: "Fine-tune bằng Tiếng Anh có nhiều dữ liệu hơn không? Tại sao không train Tiếng Anh rồi dùng Máy Dịch sang Tiếng Việt?"

Đây là câu hỏi cốt lõi mang tính định hình hướng nghiên cứu. Dưới đây là 4 luận điểm kỹ thuật và thực tiễn:

### 3.1. Thực tế dữ liệu: Sự ngộ nhận giữa X-quang và MRI Cột sống
- **Trong X-quang ngực (Chest X-ray):** Tiếng Anh thực sự áp đảo với hàng trăm nghìn báo cáo công khai (MIMIC-CXR có 227.827 báo cáo, CheXpert có 224.316 ảnh).
- **NHƯNG trong MRI Cột sống (Spine MRI): Dữ liệu báo cáo văn bản tự do (free-text clinical reports) bằng tiếng Anh gần như KHÔNG TỒN TẠI ở dạng công khai:**
  - *RSNA Lumbar Spine Degeneration Challenge (2024):* Gồm gần 2.000 ca bệnh nhưng **chỉ cung cấp nhãn phân loại (Normal, Mild, Moderate, Severe)** và tọa độ điểm (keypoints), hoàn toàn **KHÔNG có báo cáo chẩn đoán dạng văn bản tự do**.
  - *Tập dữ liệu SPIDER (2024):* Chỉ phục vụ phân đoạn (segmentation) thân đốt sống và đĩa đệm.
- $\rightarrow$ **Kết luận thực tế:** Không hề tồn tại một kho dữ liệu hàng chục nghìn báo cáo MRI cột sống tiếng Anh miễn phí nào để bạn có thể mang về pretrain. Tập dữ liệu PSPINES của bạn (247 bệnh nhân $\times$ 5 mức = 1.235 ca bệnh) thực chất là một bộ dữ liệu hiếm hoi và rất có giá trị!

---

### 3.2. Cái bẫy chết người của Pipeline "Train Tiếng Anh + Máy dịch sang Tiếng Việt"

```
[Ảnh MRI Cột Sống] 
       │
       ▼
[Model VLM Tiếng Anh]  ──── (Tiềm ẩn lỗi Hallucination 1: Sai tầng / Sai mức độ hẹp)
       │ (Sinh Text tiếng Anh)
       ▼
[Engine Dịch Máy (MT)] ──── (Tích lũy lỗi Translation Drift 2: Sai thuật ngữ giải phẫu)
       │
       ▼
[Báo cáo Tiếng Việt Cuối] ──> TỔNG HỢP LỖI LÂM SÀNG CỰC KỲ NGUY HIỂM!
```

#### 1. Lệch pha thuật ngữ giải phẫu chuyên khoa (Terminology Drift):
Các công cụ dịch tự động (Google Translate, DeepL) thường xuyên dịch sai các thuật ngữ X-quang/MRI tinh tế:
- *"dural sac compression"* $\rightarrow$ bị dịch thành *"chèn ép túi màng cứng"* (tạm được) nhưng đôi khi dịch thành *"túi cùng"* hoặc *"bao xơ"*.
- *"neural foraminal stenosis"* $\rightarrow$ dịch máy thành *"hẹp lỗ thần kinh"* (sai chuyên môn, thuật ngữ chuẩn của bác sĩ Việt Nam là *"hẹp lỗ liên hợp"*).
- *"annular tear / annular fissure"* $\rightarrow$ bị dịch nhầm thành *"rách hình khuyên"* thay vì *"rách rách vòng xơ đĩa đệm"*.
- *"Schmorl's node"* $\rightarrow$ bị dịch thành *"nút Schmorl"* hoặc bỏ sót.
- *"Modic type 1 endplate changes"* $\rightarrow$ bị dịch thành *"thay đổi tấm cuối loại 1"*.

#### 2. Tích lũy lỗi hai tầng (Cascading Error):
- Bản thân mô hình sinh ngôn ngữ thị giác (VLM) đã có xác suất dự đoán sai $p_1$.
- Bộ dịch máy lại có thêm xác suất làm méo mó ngữ nghĩa $p_2$.
- Xác suất sai hỏng tổng thể tăng vọt: $P(\text{error}) = 1 - (1 - p_1)(1 - p_2)$. Trong y khoa, chỉ cần dịch sai một từ chỉ định vị trí (ví dụ *"left"* thành *"phải"*, hoặc *"moderate"* thành *"nặng"*) sẽ dẫn tới việc chỉ định phẫu thuật sai lầm cho người bệnh!

#### 3. Xung đột cấu trúc văn phong lâm sàng (Institutional Reporting Style):
- Báo cáo tiếng Anh thường chia theo format *Findings* (tường thuật mô tả) và *Impression* (kết luận).
- Báo cáo tại các bệnh viện Việt Nam (Bạch Mai, 108, Việt Đức) có quy chuẩn hành văn rất đặc thù: Đi tuần tự theo 5 tầng $\rightarrow$ Tầng nào thoái hóa $\rightarrow$ Thoát vị thể gì (lồi, ra sau, lệch bên) $\rightarrow$ Chèn ép rễ nào $\rightarrow$ Hẹp ống sống hay hẹp lỗ liên hợp.
- Dịch máy từ tiếng Anh sang sẽ tạo ra một văn bản "ngọng nghịu", không bao giờ đạt được cấu trúc văn bản mà hội đồng bác sĩ Việt Nam chấp nhận.

---

### 3.3. Sức mạnh của Foundation Models Đa Ngôn Ngữ hiện đại (Qwen2.5 / Gemma-2)
- Các mô hình nền tảng hiện đại như **Qwen2.5-VL** hay **Gemma-2** vốn dĩ đã được tiền huấn luyện trên hàng ngàn tỷ tokens đa ngôn ngữ (bao gồm kho dữ liệu tiếng Việt phong phú và sách giáo trình y khoa).
- Bản thân mô hình **ĐÃ BIẾT TIẾNG VIỆT VÀ TỪ VỰNG Y KHOA**.
- Vai trò của quá trình tinh chỉnh (Fine-tuning với QLoRA) trên tập dữ liệu của bạn không phải là "dạy mô hình học tiếng Việt từ đầu", mà là **"Căn chỉnh phong cách hành văn chuẩn (Clinical Style Alignment) và kết nối đặc trưng ảnh lát cắt MRI với các kết luận bệnh lý tương ứng"**.
- Bằng chứng từ bài báo ViMed-PET (NeurIPS 2025): Khi fine-tune trực tiếp trên dữ liệu tiếng Việt, mô hình đạt BLEU-4 là **58.07** và ROUGE-L là **72.74**, vượt xa hoàn toàn việc dùng mô hình tiếng Anh.

---

### 3.4. Lợi thế đặc quyền của bộ dữ liệu PSPINES của bạn
Trong tệp `dataset/dataset_master.csv` và thư mục gốc của bạn:
- Cột `reports_json`: Chứa báo cáo gốc chuẩn tiếng Việt từ bác sĩ chẩn đoán hình ảnh.
- Cột `reports_text`: Chứa phiên bản tiếng Anh tương ứng.

Điều này mang lại một **lợi thế học thuật vô song**:
- Bạn hoàn toàn có thể huấn luyện mô hình sinh báo cáo tiếng Việt trực tiếp (Native Vietnamese Pipeline).
- Đồng thời, bạn có thể triển khai thực nghiệm phụ: Đánh giá mô hình trên cả tiếng Anh và tiếng Việt (Bilingual Evaluation), so sánh trực tiếp hiệu năng giữa việc sinh tiếng Việt bản địa với phương pháp qua dịch máy để đưa vào bài báo khoa học. Đây sẽ là một mục thảo luận (Discussion / Ablation Study) cực kỳ đắt giá khẳng định tính ưu việt của nghiên cứu của bạn!
