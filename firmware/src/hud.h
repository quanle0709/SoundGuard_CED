#pragma once

#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <U8g2_for_Adafruit_GFX.h>

struct HUDState {
  String subtitle;
  String environmentalSound;
  String criticalEvent;
  bool helpActive = false;
  String alert;
  String status;
};

class SoundGuardHUD {
 public:
  SoundGuardHUD(Adafruit_SSD1306& display, U8G2_FOR_ADAFRUIT_GFX& text);

  void begin();
  void setSubtitle(const String& value);
  void setPartialSubtitle(const String& value);
  void setEnvironmentalSound(const String& value);
  void setAlertState(const String& value);
  void setAlert(const String& value);
  void setStatus(const String& value);
  bool synchronizeClock(const String& value);
  void update();
  void render();
  void renderHomeScreen();
  const HUDState& state() const { return state_; }
  bool clockSynchronized() const { return clockSynchronized_; }
  char benchmarkScreenCode() const;
  String benchmarkClockText();
  uint32_t benchmarkDisplayCommitCount() const { return displayCommitCount_; }
  uint32_t benchmarkLastDisplayMicros() const { return lastDisplayMicros_; }
  uint32_t benchmarkMaxDisplayMicros() const { return maxDisplayMicros_; }

 private:
  enum class Screen { Home, Subtitle, Alert };

#ifndef HUD_OLED_WIDTH
#define HUD_OLED_WIDTH 64
#endif
#ifndef HUD_OLED_HEIGHT
#define HUD_OLED_HEIGHT 32
#endif

  // The Vietnamese U8g2 font is 8x16. It is rasterized into this buffer and
  // reduced with pixel aggregation to a readable 6x10 cell on the 64x32 HUD.
  static constexpr int kFontBufferWidth = 96;
  static constexpr int kFontBufferHeight = 16;
  static constexpr int kScaleXNumerator = 3;
  static constexpr int kScaleXDenominator = 4;
  static constexpr int kCompactLineHeight = 10;
  static constexpr int kTextWidth =
      (HUD_OLED_WIDTH * kScaleXDenominator) / kScaleXNumerator;
  static constexpr int kMaxWrappedLines = 16;
  static constexpr int kVisibleSubtitleLines = 2;
  static constexpr unsigned long kSubtitlePageDurationMs = 2000;
  static constexpr int kSubtitleFirstY = 9;
  static constexpr int kSubtitleSecondY = 20;
  static constexpr int kCedBaselineY = 6;
  static constexpr int kCedRightMargin = 1;
  static constexpr int kAlertEventBaselineY = 7;
  static constexpr int kAlertHelpBaselineY = 19;
  static constexpr int kAlertFooterBaselineY = 31;
  static constexpr int kCenteredTextMargin = 1;

  Adafruit_SSD1306& display_;
  U8G2_FOR_ADAFRUIT_GFX& text_;
  U8G2_FOR_ADAFRUIT_GFX homeText_;
  GFXcanvas1 lineBuffer_;
  HUDState state_;
  String subtitleLines_[kMaxWrappedLines];
  int subtitleLineCount_ = 0;
  int subtitlePageIndex_ = 0;
  bool subtitleIsFinal_ = false;
  unsigned long subtitlePageStartedMs_ = 0;
  char lastHomeTime_[6] = "";
  unsigned long lastHomeClockCheckMs_ = 0;
  bool clockSynchronized_ = false;
  int clockUtcOffsetMinutes_ = 0;
  uint32_t displayCommitCount_ = 0;
  uint32_t lastDisplayMicros_ = 0;
  uint32_t maxDisplayMicros_ = 0;

  bool formatCurrentTime(char (&timeBuffer)[6]);
  void commitDisplay();
  bool parseClockPayload(const String& value, int64_t& epoch,
                         int& utcOffsetMinutes);
  bool alertActive() const;
  Screen desiredScreen() const;
  void resumeSubtitleAfterAlert(bool alertWasActive);
  int wrapUTF8(const String& input, String* lines, int capacity);
  void appendWrappedToken(const String& token, String* lines, int& count,
                          int capacity);
  String fitCedLabel(const String& input, int maxWidth);
  void rebuildSubtitleLines();
  void drawCompactUTF8(int x, int y, const String& value);
  void drawCedLabelRightAligned();
  void drawCenteredText(const String& value, int baselineY,
                        const char* overflowFallback);
  void renderSubtitleScreen();
  void renderAlertScreen();
};
