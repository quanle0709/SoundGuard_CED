#include "hud.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <time.h>

#ifndef HUD_ROTATION
#define HUD_ROTATION 0
#endif

namespace {
constexpr int64_t kClockEpochMin = 946684800LL;    // 2000-01-01T00:00:00Z
constexpr int64_t kClockEpochMax = 4102444799LL;   // 2099-12-31T23:59:59Z
constexpr int kClockOffsetMinutesMin = -14 * 60;
constexpr int kClockOffsetMinutesMax = 14 * 60;

bool isUnsignedDecimal(const String& value) {
  if (value.isEmpty()) return false;
  for (unsigned int i = 0; i < value.length(); ++i) {
    if (value[i] < '0' || value[i] > '9') return false;
  }
  return true;
}

bool isSignedDecimal(const String& value) {
  if (value.isEmpty()) return false;
  unsigned int firstDigit = (value[0] == '+' || value[0] == '-') ? 1 : 0;
  if (firstDigit == value.length()) return false;
  for (unsigned int i = firstDigit; i < value.length(); ++i) {
    if (value[i] < '0' || value[i] > '9') return false;
  }
  return true;
}
}  // namespace

SoundGuardHUD::SoundGuardHUD(Adafruit_SSD1306& display,
                             U8G2_FOR_ADAFRUIT_GFX& text)
    : display_(display), text_(text),
      lineBuffer_(kFontBufferWidth, kFontBufferHeight) {}

void SoundGuardHUD::begin() {
  display_.setRotation(HUD_ROTATION);
  text_.begin(lineBuffer_);
  text_.setFontMode(1);
  text_.setForegroundColor(1);
  text_.setBackgroundColor(0);
  text_.setFont(u8g2_font_unifont_t_vietnamese1);
  homeText_.begin(display_);
  homeText_.setFontMode(1);
  homeText_.setForegroundColor(SSD1306_WHITE);
  homeText_.setBackgroundColor(SSD1306_BLACK);
  homeText_.setFont(u8g2_font_4x6_tr);
  render();
}

void SoundGuardHUD::setSubtitle(const String& value) {
  state_.subtitle = value;
  subtitleIsFinal_ = !value.isEmpty();
  subtitlePageIndex_ = 0;
  subtitlePageStartedMs_ = millis();
  rebuildSubtitleLines();
  if (!alertActive()) render();
}

void SoundGuardHUD::setPartialSubtitle(const String& value) {
  state_.subtitle = value;
  subtitleIsFinal_ = false;
  subtitlePageIndex_ = 0;
  rebuildSubtitleLines();
  if (!alertActive()) render();
}

void SoundGuardHUD::setEnvironmentalSound(const String& value) {
  state_.environmentalSound = value;
  if (!alertActive()) render();
}

bool SoundGuardHUD::alertActive() const {
  return !state_.criticalEvent.isEmpty() || state_.helpActive ||
         !state_.alert.isEmpty();
}

SoundGuardHUD::Screen SoundGuardHUD::desiredScreen() const {
  if (alertActive()) return Screen::Alert;
  if (!state_.subtitle.isEmpty() || !state_.environmentalSound.isEmpty()) {
    return Screen::Subtitle;
  }
  return Screen::Home;
}

char SoundGuardHUD::benchmarkScreenCode() const {
  switch (desiredScreen()) {
    case Screen::Alert:
      return 'A';
    case Screen::Subtitle:
      return 'S';
    case Screen::Home:
      return 'H';
  }
  return '?';
}

String SoundGuardHUD::benchmarkClockText() {
  char timeBuffer[6];
  formatCurrentTime(timeBuffer);
  return String(timeBuffer);
}

void SoundGuardHUD::resumeSubtitleAfterAlert(bool alertWasActive) {
  if (alertWasActive && !alertActive() && subtitleIsFinal_) {
    // The ALERT layer pauses pagination; the interrupted subtitle page gets
    // its complete reading interval when Screen 2 becomes visible again.
    subtitlePageStartedMs_ = millis();
  }
}

void SoundGuardHUD::setAlertState(const String& value) {
  const bool alertWasActive = alertActive();
  if (!value.isEmpty()) {
    const bool validHelpFlag = value[0] == '0' || value[0] == '1';
    const bool validSeparator = value.length() >= 2 && value[1] == '|';
    const bool hasCondition = value[0] == '1' || value.length() > 2;
    if (!validHelpFlag || !validSeparator || !hasCondition) return;
  }
  state_.alert = "";
  if (value.isEmpty()) {
    state_.helpActive = false;
    state_.criticalEvent = "";
  } else {
    state_.helpActive = value[0] == '1';
    state_.criticalEvent = value.length() > 2 ? value.substring(2) : "";
  }
  resumeSubtitleAfterAlert(alertWasActive);
  render();
}

void SoundGuardHUD::setAlert(const String& value) {
  const bool alertWasActive = alertActive();
  state_.criticalEvent = "";
  state_.helpActive = false;
  state_.alert = value;
  resumeSubtitleAfterAlert(alertWasActive);
  render();
}

void SoundGuardHUD::setStatus(const String& value) {
  state_.status = value;
}

bool SoundGuardHUD::parseClockPayload(const String& value, int64_t& epoch,
                                      int& utcOffsetMinutes) {
  const int separator = value.indexOf(',');
  if (separator <= 0 || separator >= static_cast<int>(value.length()) - 1 ||
      value.indexOf(',', separator + 1) >= 0) {
    return false;
  }

  const String epochText = value.substring(0, separator);
  const String offsetText = value.substring(separator + 1);
  if (!isUnsignedDecimal(epochText) || !isSignedDecimal(offsetText)) {
    return false;
  }

  errno = 0;
  char* epochEnd = nullptr;
  const long long parsedEpoch = strtoll(epochText.c_str(), &epochEnd, 10);
  if (errno == ERANGE || epochEnd == nullptr || *epochEnd != '\0' ||
      parsedEpoch < kClockEpochMin || parsedEpoch > kClockEpochMax) {
    return false;
  }

  errno = 0;
  char* offsetEnd = nullptr;
  const long parsedOffset = strtol(offsetText.c_str(), &offsetEnd, 10);
  if (errno == ERANGE || offsetEnd == nullptr || *offsetEnd != '\0' ||
      parsedOffset < kClockOffsetMinutesMin ||
      parsedOffset > kClockOffsetMinutesMax) {
    return false;
  }

  epoch = static_cast<int64_t>(parsedEpoch);
  utcOffsetMinutes = static_cast<int>(parsedOffset);
  return true;
}

bool SoundGuardHUD::synchronizeClock(const String& value) {
  int64_t epoch = 0;
  int utcOffsetMinutes = 0;
  if (!parseClockPayload(value, epoch, utcOffsetMinutes)) return false;

  const time_t utcSeconds = static_cast<time_t>(epoch);
  if (static_cast<int64_t>(utcSeconds) != epoch) return false;

  char previousTime[6];
  const bool homeWasVisible = desiredScreen() == Screen::Home;
  const bool previouslySynchronized = clockSynchronized_;
  if (homeWasVisible) formatCurrentTime(previousTime);

  timeval clockValue = {utcSeconds, 0};
  if (settimeofday(&clockValue, nullptr) != 0) return false;

  clockUtcOffsetMinutes_ = utcOffsetMinutes;
  clockSynchronized_ = true;

  if (homeWasVisible) {
    char synchronizedTime[6];
    formatCurrentTime(synchronizedTime);
    if (!previouslySynchronized ||
        std::memcmp(previousTime, synchronizedTime,
                    sizeof(synchronizedTime)) != 0) {
      renderHomeScreen();
    }
  }
  return true;
}

bool SoundGuardHUD::formatCurrentTime(char (&timeBuffer)[6]) {
  if (!clockSynchronized_) {
    std::memcpy(timeBuffer, "--:--", sizeof(timeBuffer));
    return false;
  }

  const time_t utcNow = time(nullptr);
  const int64_t localEpochValue =
      static_cast<int64_t>(utcNow) +
      static_cast<int64_t>(clockUtcOffsetMinutes_) * 60;
  const time_t localEpoch = static_cast<time_t>(localEpochValue);
  tm localTime;
  if (static_cast<int64_t>(localEpoch) != localEpochValue ||
      gmtime_r(&localEpoch, &localTime) == nullptr) {
    std::memcpy(timeBuffer, "--:--", sizeof(timeBuffer));
    return false;
  }

  snprintf(timeBuffer, sizeof(timeBuffer), "%02d:%02d", localTime.tm_hour,
           localTime.tm_min);
  return true;
}

void SoundGuardHUD::commitDisplay() {
  const uint32_t startedMicros = micros();
  display_.display();
  lastDisplayMicros_ = micros() - startedMicros;
  if (lastDisplayMicros_ > maxDisplayMicros_) {
    maxDisplayMicros_ = lastDisplayMicros_;
  }
  ++displayCommitCount_;
}

void SoundGuardHUD::renderHomeScreen() {
  char timeBuffer[6];
  formatCurrentTime(timeBuffer);

  display_.clearDisplay();
  homeText_.setFont(u8g2_font_4x6_tr);
  homeText_.drawStr(0, 5, "soundguard");
  homeText_.drawStr(0, 32, timeBuffer);
  homeText_.drawStr(44, 31, "Ready");
  commitDisplay();

  std::memcpy(lastHomeTime_, timeBuffer, sizeof(lastHomeTime_));
}

void SoundGuardHUD::update() {
  const unsigned long now = millis();
  if (alertActive()) return;

  if (subtitleIsFinal_ && !state_.subtitle.isEmpty()) {
    if (now - subtitlePageStartedMs_ < kSubtitlePageDurationMs) return;
    const int pageCount =
        (subtitleLineCount_ + kVisibleSubtitleLines - 1) /
        kVisibleSubtitleLines;
    if (subtitlePageIndex_ + 1 < pageCount) {
      ++subtitlePageIndex_;
      subtitlePageStartedMs_ = now;
    } else {
      state_.subtitle = "";
      subtitleLineCount_ = 0;
      subtitlePageIndex_ = 0;
      subtitleIsFinal_ = false;
    }
    render();
    return;
  }

  if (!state_.subtitle.isEmpty() || !state_.environmentalSound.isEmpty()) {
    return;
  }

  if (now - lastHomeClockCheckMs_ < 1000) return;
  lastHomeClockCheckMs_ = now;

  char timeBuffer[6];
  formatCurrentTime(timeBuffer);
  if (std::memcmp(timeBuffer, lastHomeTime_, sizeof(timeBuffer)) != 0) {
    renderHomeScreen();
  }
}

void SoundGuardHUD::appendWrappedToken(const String& token, String* lines,
                                       int& count, int capacity) {
  if (token.isEmpty() || count >= capacity) return;
  String remaining = token;
  while (!remaining.isEmpty() && count < capacity) {
    String piece;
    int byteEnd = 0;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(remaining.c_str());
    while (byteEnd < static_cast<int>(remaining.length())) {
      int charBytes = 1;
      if ((bytes[byteEnd] & 0xE0) == 0xC0) charBytes = 2;
      else if ((bytes[byteEnd] & 0xF0) == 0xE0) charBytes = 3;
      else if ((bytes[byteEnd] & 0xF8) == 0xF0) charBytes = 4;
      String candidate = remaining.substring(0, byteEnd + charBytes);
      if (text_.getUTF8Width(candidate.c_str()) > kTextWidth) break;
      piece = candidate;
      byteEnd += charBytes;
    }
    if (piece.isEmpty()) {
      // Invalid or unexpectedly wide input: consume one complete UTF-8 sequence.
      int charBytes = 1;
      if ((bytes[0] & 0xE0) == 0xC0) charBytes = 2;
      else if ((bytes[0] & 0xF0) == 0xE0) charBytes = 3;
      else if ((bytes[0] & 0xF8) == 0xF0) charBytes = 4;
      piece = remaining.substring(0, charBytes);
      byteEnd = charBytes;
    }
    lines[count++] = piece;
    remaining = remaining.substring(byteEnd);
  }
}

int SoundGuardHUD::wrapUTF8(const String& input, String* lines, int capacity) {
  int count = 0;
  int start = 0;
  String current;
  while (start < static_cast<int>(input.length()) && count < capacity) {
    while (start < static_cast<int>(input.length()) &&
           (input[start] == ' ' || input[start] == '\r')) ++start;
    if (start >= static_cast<int>(input.length())) break;
    if (input[start] == '\n') {
      if (!current.isEmpty() && count < capacity) lines[count++] = current;
      current = "";
      ++start;
      continue;
    }
    int space = input.indexOf(' ', start);
    int newline = input.indexOf('\n', start);
    int end = space;
    if (end < 0 || (newline >= 0 && newline < end)) end = newline;
    if (end < 0) end = input.length();
    String word = input.substring(start, end);
    String candidate = current.isEmpty() ? word : current + " " + word;
    if (text_.getUTF8Width(candidate.c_str()) <= kTextWidth) {
      current = candidate;
    } else {
      if (!current.isEmpty()) lines[count++] = current;
      current = "";
      if (text_.getUTF8Width(word.c_str()) <= kTextWidth) {
        current = word;
      } else {
        appendWrappedToken(word, lines, count, capacity);
      }
    }
    start = end;
  }
  if (!current.isEmpty() && count < capacity) lines[count++] = current;
  return count;
}

void SoundGuardHUD::rebuildSubtitleLines() {
  for (int i = 0; i < kMaxWrappedLines; ++i) subtitleLines_[i] = "";
  subtitleLineCount_ =
      wrapUTF8(state_.subtitle, subtitleLines_, kMaxWrappedLines);
}

void SoundGuardHUD::drawCompactUTF8(int x, int y, const String& value) {
  lineBuffer_.fillScreen(0);
  text_.drawUTF8(0, 14, value.c_str());

  int sourceWidth =
      min(static_cast<int>(text_.getUTF8Width(value.c_str())),
          kFontBufferWidth);
  int outputWidth =
      (sourceWidth * kScaleXNumerator + kScaleXDenominator - 1) /
      kScaleXDenominator;
  for (int destinationY = 0; destinationY < kCompactLineHeight;
       ++destinationY) {
    int sourceYStart = destinationY * kFontBufferHeight /
                       kCompactLineHeight;
    int sourceYEnd = ((destinationY + 1) * kFontBufferHeight +
                      kCompactLineHeight - 1) /
                     kCompactLineHeight;
    for (int destinationX = 0; destinationX < outputWidth; ++destinationX) {
      int sourceXStart = destinationX * kScaleXDenominator /
                         kScaleXNumerator;
      int sourceXEnd = ((destinationX + 1) * kScaleXDenominator +
                        kScaleXNumerator - 1) /
                       kScaleXNumerator;
      bool pixelSet = false;
      for (int sourceY = sourceYStart; sourceY < sourceYEnd && !pixelSet;
           ++sourceY) {
        for (int sourceX = sourceXStart; sourceX < sourceXEnd; ++sourceX) {
          if (lineBuffer_.getPixel(sourceX, sourceY)) {
            pixelSet = true;
            break;
          }
        }
      }
      const int screenX = x + destinationX;
      const int screenY = y + destinationY;
      if (pixelSet && screenX >= 0 && screenX < HUD_OLED_WIDTH &&
          screenY >= 0 && screenY < HUD_OLED_HEIGHT) {
        display_.drawPixel(screenX, screenY, SSD1306_WHITE);
      }
    }
  }
}

String SoundGuardHUD::fitCedLabel(const String& input, int maxWidth) {
  String result;
  int byteEnd = 0;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(input.c_str());
  while (byteEnd < static_cast<int>(input.length())) {
    int charBytes = 1;
    if ((bytes[byteEnd] & 0xE0) == 0xC0) charBytes = 2;
    else if ((bytes[byteEnd] & 0xF0) == 0xE0) charBytes = 3;
    else if ((bytes[byteEnd] & 0xF8) == 0xF0) charBytes = 4;
    String candidate = input.substring(0, byteEnd + charBytes);
    if (homeText_.getUTF8Width(candidate.c_str()) > maxWidth) break;
    result = candidate;
    byteEnd += charBytes;
  }
  return result;
}

void SoundGuardHUD::drawCedLabelRightAligned() {
  if (state_.environmentalSound.isEmpty()) return;
  homeText_.setFont(u8g2_font_4x6_tr);
  const int maxWidth = HUD_OLED_WIDTH - (2 * kCedRightMargin);
  const String label = fitCedLabel(state_.environmentalSound, maxWidth);
  const int width = homeText_.getUTF8Width(label.c_str());
  const int x = max(kCedRightMargin,
                    HUD_OLED_WIDTH - kCedRightMargin - width);
  homeText_.drawUTF8(x, kCedBaselineY, label.c_str());
}

void SoundGuardHUD::drawCenteredText(const String& value, int baselineY,
                                     const char* overflowFallback) {
  homeText_.setFont(u8g2_font_4x6_tr);
  const int maxWidth = HUD_OLED_WIDTH - (2 * kCenteredTextMargin);
  String visible = value;
  if (homeText_.getUTF8Width(visible.c_str()) > maxWidth) {
    // Structured runtime labels are compact mapped categories. EVENT is an
    // unambiguous safety fallback for malformed/legacy oversized payloads.
    visible = overflowFallback;
  }
  const int width = homeText_.getUTF8Width(visible.c_str());
  const int x = max(kCenteredTextMargin, (HUD_OLED_WIDTH - width) / 2);
  homeText_.drawUTF8(x, baselineY, visible.c_str());
}

void SoundGuardHUD::renderSubtitleScreen() {
  display_.clearDisplay();
  drawCedLabelRightAligned();

  int first = 0;
  if (subtitleIsFinal_) {
    first = subtitlePageIndex_ * kVisibleSubtitleLines;
  } else {
    // Partial results stay responsive and stable: show the newest viewport,
    // without starting or cycling the finalized-utterance slideshow.
    first = max(0, subtitleLineCount_ - kVisibleSubtitleLines);
  }
  if (first < subtitleLineCount_) {
    drawCompactUTF8(0, kSubtitleFirstY, subtitleLines_[first]);
  }
  if (first + 1 < subtitleLineCount_) {
    drawCompactUTF8(0, kSubtitleSecondY, subtitleLines_[first + 1]);
  }
  commitDisplay();
}

void SoundGuardHUD::renderAlertScreen() {
  display_.clearDisplay();
  const String eventLabel = !state_.criticalEvent.isEmpty()
                                ? state_.criticalEvent
                                : (state_.helpActive ? "SOS" : "ALERT");
  drawCenteredText(eventLabel, kAlertEventBaselineY, "EVENT");
  if (state_.helpActive) {
    drawCenteredText("HELP", kAlertHelpBaselineY, "HELP");
  }
  drawCenteredText("! ALERT !", kAlertFooterBaselineY, "ALERT");
  commitDisplay();
}

void SoundGuardHUD::render() {
  switch (desiredScreen()) {
    case Screen::Alert:
      renderAlertScreen();
      break;
    case Screen::Subtitle:
      renderSubtitleScreen();
      break;
    case Screen::Home:
      renderHomeScreen();
      break;
  }
}
