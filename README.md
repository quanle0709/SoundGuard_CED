# SoundGuard CED

## Improvement modes (default remains legacy)

The safety-improvement adapters are explicit opt-ins. With both variables absent, CED and personalization retain the frozen legacy behavior.

Legacy mode (DTLN remains enabled for STT as before):

```powershell
Remove-Item Env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING -ErrorAction SilentlyContinue
Remove-Item Env:SOUNDGUARD_SAFE_COMPOUND_EVIDENCE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe app.py --mic --duration 5
```

Improved development mode uses the audited coarse CED taxonomy, compound-safe personalization, raw CED, and raw STT. DTLN remains available as a standalone filter:

```powershell
$env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING='1'
$env:SOUNDGUARD_SAFE_COMPOUND_EVIDENCE='1'
.\.venv\Scripts\python.exe app.py --mic --duration 5 --no-dtln
```

Rerun the frozen improved development benchmark without overwriting baseline evidence:

```powershell
$env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING='1'
$env:SOUNDGUARD_SAFE_COMPOUND_EVIDENCE='1'
.\.venv\Scripts\python.exe -m benchmark.run_improved_benchmark
```

The measured development result and limitations are in `benchmark_results/improved/FINAL_IMPROVEMENT_REPORT.md`. Emergency recall remains below the safety target; improved mode is not final holdout validation.

## Emergency V3 EfficientSED specialist (opt-in)

Emergency V3 is an additional PC-side safety-evidence branch. It does not replace CED-Tiny, does not rename normal CED output, and is disabled by default. The isolated setup uses only the audited EfficientSED repository:

    git clone --depth 1 https://github.com/theMoro/EfficientSED.git benchmark_data\external\efficientsed_repo
    .\.venv\Scripts\python.exe -m venv benchmark_data\external\efficientsed_venv
    benchmark_data\external\efficientsed_venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    benchmark_data\external\efficientsed_venv\Scripts\python.exe -m pip install librosa
    benchmark_data\external\efficientsed_venv\Scripts\python.exe benchmark\experiments\emergency_v3\efficientsed_adapter.py --download

Run the improved V2 path with Emergency V3 disabled:

    Remove-Item Env:SOUNDGUARD_ENABLE_EMERGENCY_V3 -ErrorAction SilentlyContinue
    $env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING='1'
    .\.venv\Scripts\python.exe app.py --mic --duration 5 --no-dtln

Enable Emergency V3 explicitly:

    $env:SOUNDGUARD_ENABLE_SEMANTIC_CED_MAPPING='1'
    .\.venv\Scripts\python.exe app.py --mic --duration 5 --no-dtln --emergency-v3

Alternatively set SOUNDGUARD_ENABLE_EMERGENCY_V3=1 and omit the CLI switch. The first audio window lazily loads the isolated model; later windows reuse it. Missing model/dependencies or inference failure logs once and falls back to V2.

Rerun the Emergency V3 benchmark and plots:

    benchmark_data\external\efficientsed_venv\Scripts\python.exe benchmark\experiments\emergency_v3\run_evaluation.py
    .\.venv\Scripts\python.exe benchmark\experiments\emergency_v3\postprocess_results.py

The development evidence, limitations, recovery instructions, and architecture are under benchmark_results/emergency_v3.

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

## OLED HUD trên kính SoundGuard

HUD giữ nguyên phụ đề và cảnh báo gần nhất trên OLED 64x32 thay vì xóa
màn hình giữa các bản tin. Phần mềm máy tính gửi UTF-8 đã chuẩn hóa NFC qua một
khung serial có độ dài và CRC; firmware chỉ cập nhật trạng thái sau khi nhận đủ
khung hợp lệ. Phụ đề được đo và xuống dòng theo pixel bằng
`U8g2_for_Adafruit_GFX`, với `u8g2_font_unifont_t_vietnamese1` được kết xuất
thành ô 6x10 dễ đọc, không theo số byte hoặc số ký tự cố định. Hai dòng phụ đề
chiếm 22 pixel phía trên; cảnh báo một dòng chiếm 10 pixel cuối màn hình. HUD
không có đồng hồ, thanh trạng thái, đường viền hoặc trang trí.

HOME clock synchronization uses the existing CRC-protected serial protocol. The
host sends frame type `K` with ASCII payload `<utc_epoch_seconds>,<utc_offset_minutes>`
when the HUD port opens, after a reconnect, and approximately every 10 minutes.
Both values come from the host OS at runtime. The ESP32 sets its system clock to
UTC, retains the supplied offset, and applies that offset exactly once when HOME
formats `HH:MM`; before the first valid frame it displays `--:--`.

Firmware PlatformIO nằm trong `firmware/`. Trước khi flash, sửa `HUD_OLED_SDA`,
`HUD_OLED_SCL`, địa chỉ I2C và `HUD_ROTATION` trong `firmware/platformio.ini`
cho đúng dây nối và quang học của kính. Cấu hình ESP32 DevKit V1 mặc định là
SDA GPIO 21, SCL GPIO 22, địa
chỉ `0x3C`, hướng bình thường; `HUD_ROTATION=2` xoay 180 độ mà không đảo byte
UTF-8.

Với VS Code, cài extension PlatformIO IDE, mở thư mục `firmware`, chọn
environment `esp32dev`, rồi chạy **PlatformIO: Upload**. Tương đương
trên terminal:

```powershell
cd .\firmware
pio run
pio run --target upload --upload-port COM5
cd ..
```

Đóng PlatformIO Serial Monitor trước khi Python mở cùng cổng COM. Chạy bộ màn
hình kiểm tra tiếng Việt xác định (mỗi mẫu hiển thị một giây):

```powershell
python app.py --hud-port COM5 --hud-test
```

Chạy pipeline hiện có cùng HUD:

```powershell
python app.py --live-stt --hud-port COM5
python app.py --continuous --duration 5 --hud-port COM5
python app.py ".\path\to\audio.wav" --hud-port COM5
```

Không truyền `--hud-port` thì mọi chế độ cũ tiếp tục chỉ ghi ra terminal như
trước. Với ESP32 DevKit V1, dùng cổng USB-to-UART của board và đóng Serial
Monitor trước khi Python mở cùng cổng COM.

Các software test không cần microphone hoặc gọi model/network bên ngoài:

```powershell
.\run_software_tests.ps1
```

## AI personalized alert priorities (MVP)

The personalization assistant is isolated from CED, STT, filtering, and HUD
transport. Start its local web interface with:

```powershell
.\.venv\Scripts\python.exe -m personalization.web_server
```

Open `http://127.0.0.1:8765`, complete the short interview, review the generated
1–5 profile, then select **Apply to SoundGuard**. OpenAI is the default provider
and uses `gpt-5-mini`. Configure it with `OPENAI_API_KEY`. To use OpenRouter and
its default `openai/gpt-oss-120b:free` model instead, start the server in the
same PowerShell session with:

```powershell
$env:SOUNDGUARD_AI_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
Remove-Item Env:SOUNDGUARD_AI_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m personalization.web_server
```

`SOUNDGUARD_AI_MODEL` overrides the model for either provider. For example:

```powershell
$env:SOUNDGUARD_AI_MODEL = "openai/gpt-oss-120b:free"
```

Both providers request the existing strict JSON schema and pass responses
through the same safety validator. If configuration, API, network, schema, or
response handling fails, the same flow uses the classified deterministic
multi-role fallback. The result page and server terminal print the provider;
successful OpenRouter generation is shown as `openrouter`, while fallback is
shown as `deterministic_fallback:<reason>`.

Enable the thin runtime adapter when starting SoundGuard:

```powershell
python app.py --continuous --duration 5 --personalized-alerts
```

The adapter reloads an applied profile without allowing AI output to address
the OLED, vibration motors, or other hardware. Omitting `--personalized-alerts`
preserves the original alert behavior. A missing, malformed, or unsupported
profile falls back to `personalization/default_profile.json`.

Run the isolated personalization checks with:

```powershell
.\.venv\Scripts\python.exe test_personalization.py
```

Run the reproducible PC benchmark suite, including the explicitly enabled
Google STT calls, with:

```powershell
.\.venv\Scripts\python.exe benchmark\run_full_benchmark.py --enable-network-stt --stability-seconds 60
```

Use `--resume` to keep completed checkpoints after an interruption. Results,
figures, methodology, limitations, and the final report are written under
`benchmark_results/`. The included 60-second stability run is a bounded
software smoke soak; use `--stability-seconds 3600` for the one-hour protocol.

The expanded public-dataset suite preserves those preliminary results and
writes separately to `benchmark_results/expanded/`. Acquire its bounded data
and run the full protocol with:

```powershell
.\.venv\Scripts\python.exe benchmark\download_datasets.py
.\.venv\Scripts\python.exe benchmark\run_full_benchmark.py --expanded --enable-network-stt --stability-seconds 3600 --resume
```

Downloaded ESC-50, VIVOS, and DEMAND binaries plus generated mixtures are
gitignored. Sources, licenses, selected portions, denominators, and leakage
caveats are documented in `benchmark_data/DATASETS.md` and
`benchmark_results/expanded/dataset_leakage_audit.md`.

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
