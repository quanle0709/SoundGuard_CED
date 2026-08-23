#pragma once

#include <Arduino.h>

#include "hud.h"

class HUDProtocolParser {
 public:
  explicit HUDProtocolParser(SoundGuardHUD& hud) : hud_(hud) {}
  void poll(Stream& input, Print* benchmarkOutput = nullptr);

 private:
  static constexpr uint16_t kMaxPayload = 1024;
  enum class State : uint8_t {
    MagicS, MagicG, Version, Type, LengthHigh, LengthLow, Payload, CrcHigh, CrcLow
  };

  SoundGuardHUD& hud_;
  State state_ = State::MagicS;
  uint8_t type_ = 0;
  uint16_t length_ = 0;
  uint16_t received_ = 0;
  uint16_t crc_ = 0xFFFF;
  uint16_t receivedCrc_ = 0;
  String payload_;
  Print* benchmarkOutput_ = nullptr;
  uint32_t rxBytes_ = 0;
  uint32_t completedFrames_ = 0;
  uint32_t validFrames_ = 0;
  uint32_t crcFailures_ = 0;
  uint32_t parserRecoveries_ = 0;
  uint32_t malformedBenchmarkFrames_ = 0;
  uint32_t ackAttempts_ = 0;
  uint32_t ackBytesAccepted_ = 0;
  uint32_t ackShortWrites_ = 0;
  uint32_t pollCount_ = 0;
  int minTxAvailable_ = -1;
  String lastParsedTrialId_;
  String lastAckAttemptTrialId_;

  static uint16_t updateCrc(uint16_t crc, uint8_t value);
  void consume(uint8_t value);
  void reset(uint8_t possibleMagic = 0, bool recovery = false);
  void applyFrame();
  void applyProductFrame(uint8_t type, const String& payload);
  void applyBenchmarkFrame(Print& output);
  void emitBenchmarkDiagnostics(Print& output, const String& trialId);
};
