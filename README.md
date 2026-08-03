# SoundGuard CED

SoundGuard CED là nguyên mẫu phần mềm hỗ trợ người khiếm thính bằng cách phân tích âm thanh môi trường, chuyển giọng nói thành văn bản, phát hiện tình huống khẩn cấp và tăng cường tiếng nói trong nhiễu. Dự án đang ở trạng thái prototype; benchmark và tích hợp trên thiết bị thực vẫn đang tiếp tục. Các kết quả hiện có không đủ để tuyên bố độ chính xác trong thực tế.

## Chức năng chính

1. **Environmental Sound Detection** — phân loại âm thanh bằng CED-Tiny (`mispeech/ced-tiny`).
2. **Live Speech-to-Text** — gom câu nói bằng Silero VAD, tùy chọn lọc nhiễu DTLN, rồi nhận dạng tiếng Việt qua Google Speech Recognition.
3. **Emergency Detection** — kết hợp nhãn âm thanh, transcript và ngữ cảnh để tạo mức cảnh báo.
4. **Noise Filtering / Speech Enhancement** — chạy hai tầng DTLN TensorFlow Lite trên nhánh tiếng nói; âm thanh thô dành cho CED không bị thay thế.

## Kiến trúc

```text
WAV file / microphone
        |
        +--> raw audio --> CED-Tiny --> sound evidence --------+
        |                                                       |
        +--> Silero VAD --> DTLN --> Google STT --> text evidence
                                                                |
                                      EmergencySystem + fusion -+
                                                |
                                          alert mapping/output
```

Trong chế độ microphone, một `sounddevice.InputStream` duy nhất cấp frame cho các queue hữu hạn độc lập. Nhánh CED dùng cửa sổ âm thanh thô; nhánh lời nói dùng VAD, pre-roll, end-silence, post-roll và snapshot câu nói hoàn chỉnh.

## Module chính

- `app.py`: CLI và điều phối các chế độ file/microphone.
- `sound_classifier.py`: tải và chạy CED-Tiny qua Transformers.
- `audio_capture.py`, `streaming_audio.py`, `audio_pipeline.py`: thu và phân phối audio.
- `live_speech_to_text.py`, `speech_recognizer.py`: phân đoạn câu và Google STT (`vi-VN`).
- `voice_activity_detector.py`: Silero VAD, gồm xử lý frame streaming.
- `speech_enhancer.py`: DTLN TensorFlow Lite hai tầng.
- `emergency_system.py`, `fusion_engine.py`, `alert_mapper.py`: luật khẩn cấp, hợp nhất bằng chứng và thông báo.
- `benchmark_runner.py`, `benchmark_metrics.py`, `benchmark_report.py`: benchmark file cố định và báo cáo.

## Cài đặt trên Windows

Yêu cầu Python phù hợp với các package trong `requirements.txt` và thiết bị audio tương thích với `sounddevice` nếu dùng microphone.

```powershell
cd path\to\SoundGuard_CED
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Model

- CED-Tiny được khai báo bằng model ID `mispeech/ced-tiny`. `transformers.pipeline` tải model từ Hugging Face ở lần dùng đầu tiên nếu cache cục bộ chưa có; trọng số tải về không thuộc repository.
- Silero VAD được nạp qua package `silero-vad` ở lần dùng đầu tiên.
- DTLN dùng `external/DTLN-master/pretrained_model/model_1.tflite` và `model_2.tflite`. Trọng số model bị loại khỏi Git. Cần lấy model từ dự án DTLN upstream và đặt đúng hai đường dẫn trên trước khi bật DTLN.

## Cách chạy được CLI hỗ trợ

Phân tích một WAV có sẵn:

```powershell
python app.py ".\path\to\audio.wav"
```

Thu một lần, nghe liên tục, hoặc live STT:

```powershell
python app.py --mic --duration 8
python app.py --continuous --duration 5 --pause 1
python app.py --live-stt --end-silence-ms 700 --post-roll-ms 500
```

Liệt kê/chọn thiết bị và tùy chỉnh pipeline:

```powershell
python app.py --list-devices
python app.py --mic --duration 8 --device 1
python app.py --mic --duration 8 --no-dtln
python app.py ".\path\to\audio.wav" --no-vad --context neutral
```

Xem toàn bộ tùy chọn hiện hành, gồm ngưỡng VAD, pre/post-roll, kích thước queue và overlap CED:

```powershell
python app.py --help
```

`--duration` là thời gian chờ tối đa trong chế độ thu một lần và độ dài cửa sổ CED trong chế độ liên tục. `--ced-overlap-seconds` mặc định bằng 0; các cửa sổ chồng lấn là quan sát tương quan và có thể ảnh hưởng luật bỏ phiếu 2-trên-3.

## Test và benchmark

Các software test không cần microphone hoặc gọi model/network bên ngoài:

```powershell
.\run_software_tests.ps1
```

Benchmark offline không bật Google STT:

```powershell
python benchmark_runner.py --run-id baseline_01
python benchmark_report.py --run-id baseline_01
```

Chỉ bật cuộc gọi Google STT khi đã có sự đồng ý và kết nối mạng:

```powershell
python benchmark_runner.py --run-id baseline_stt_01 --enable-stt
```

Xem `benchmark_v1/README.md` và `benchmark_v1/MANUAL_TEST_CHECKLIST.md` để biết schema và quy trình đánh giá.

## Giới hạn và quyền riêng tư

- Google STT cần Internet và gửi audio của nhánh lời nói đến dịch vụ bên ngoài; chỉ sử dụng khi người dùng đã hiểu và đồng ý.
- Chế độ microphone có thể xử lý âm thanh riêng tư. Không commit bản ghi, thư mục `test_outputs`, hoặc sample chưa xác nhận quyền phân phối.
- Các WAV trong `samples/` hiện bị loại khỏi Git trong khi chờ xác nhận nguồn, quyền riêng tư và quyền tái phân phối.
- Model CED/Silero có thể cần tải ở lần chạy đầu; DTLN cần trọng số cục bộ không được đưa vào Git.
- Benchmark file cố định không kiểm tra phần cứng microphone, độ trễ thiết bị hoặc biến thiên môi trường thực.
- Tập benchmark hiện nhỏ và có đường dẫn đến audio cục bộ bị loại khỏi repository, nên người dùng mới cần tự cung cấp dữ liệu hợp lệ.

## Mã và phụ thuộc bên thứ ba

DTLN do Nils L. Westhausen phát triển. `external/DTLN-master/README.md` và `external/DTLN-master/LICENSE` được giữ từ upstream để ghi nguồn và giấy phép MIT của DTLN. Runtime SoundGuard dùng phần tích hợp trong `speech_enhancer.py`; các script training, evaluation, conversion và real-time demo của upstream không cần cho bốn chức năng nên không nằm trong tập public đã thu gọn. DTLN là thành phần bên thứ ba, không phải do nhóm SoundGuard tự viết. Repository này không tự cấp giấy phép nguồn mở cho phần mã SoundGuard còn lại. Các thư viện Python khác tuân theo giấy phép và điều khoản riêng của từng dự án/model/dịch vụ.
