#include "protocol.h"

#include <ESP.h>
#include <esp_system.h>

uint16_t HUDProtocolParser::updateCrc(uint16_t crc, uint8_t value) {
  crc ^= static_cast<uint16_t>(value) << 8;
  for (uint8_t i = 0; i < 8; ++i) {
    crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                         : static_cast<uint16_t>(crc << 1);
  }
  return crc;
}

void HUDProtocolParser::reset(uint8_t possibleMagic, bool recovery) {
  if (recovery) ++parserRecoveries_;
  state_ = possibleMagic == 'S' ? State::MagicG : State::MagicS;
  type_ = 0;
  length_ = received_ = receivedCrc_ = 0;
  crc_ = 0xFFFF;
  payload_ = "";
}

void HUDProtocolParser::applyProductFrame(uint8_t type, const String& payload) {
  if (type == 'S') hud_.setSubtitle(payload);
  else if (type == 'P') hud_.setPartialSubtitle(payload);
  else if (type == 'A') hud_.setAlert(payload);
  else if (type == 'E') hud_.setAlertState(payload);
  else if (type == 'C') hud_.setEnvironmentalSound(payload);
  else if (type == 'T') hud_.setStatus(payload);
  else if (type == 'K') hud_.synchronizeClock(payload);
}

void HUDProtocolParser::applyBenchmarkFrame(Print& output) {
  const int firstSeparator = payload_.indexOf('|');
  const int secondSeparator = payload_.indexOf('|', firstSeparator + 1);
  if (firstSeparator <= 0 || secondSeparator != firstSeparator + 2) {
    ++malformedBenchmarkFrames_;
    return;
  }

  const String trialId = payload_.substring(0, firstSeparator);
  if (trialId.length() > 40) {
    ++malformedBenchmarkFrames_;
    return;
  }
  for (unsigned int index = 0; index < trialId.length(); ++index) {
    const char value = trialId[index];
    if (!isAlphaNumeric(value) && value != '-' && value != '_') {
      ++malformedBenchmarkFrames_;
      return;
    }
  }

  const uint8_t innerType = payload_[firstSeparator + 1];
  const String innerPayload = payload_.substring(secondSeparator + 1);
  lastParsedTrialId_ = trialId;
  if (innerType != 'Q' && innerType != 'D') {
    if (innerType != 'S' && innerType != 'P' && innerType != 'A' &&
        innerType != 'E' && innerType != 'C' && innerType != 'T' &&
        innerType != 'K') {
      ++malformedBenchmarkFrames_;
      return;
    }
    // Product setters synchronously call display_.display() when rendering is
    // required. The ACK is deliberately emitted only after that call returns.
    applyProductFrame(innerType, innerPayload);
  }

  const int txAvailable = output.availableForWrite();
  if (minTxAvailable_ < 0 || txAvailable < minTxAvailable_) {
    minTxAvailable_ = txAvailable;
  }
  ++ackAttempts_;
  lastAckAttemptTrialId_ = trialId;
  const String clockText = hud_.benchmarkClockText();
  char ackLine[96];
  const int formatted = snprintf(
      ackLine, sizeof(ackLine), "SGACK|%s|%c|%c|%c|%s|%lu|1\r\n",
      trialId.c_str(), innerType, hud_.benchmarkScreenCode(),
      hud_.clockSynchronized() ? '1' : '0', clockText.c_str(), millis());
  const size_t expected =
      formatted > 0 && formatted < static_cast<int>(sizeof(ackLine))
          ? static_cast<size_t>(formatted)
          : 0;
  const size_t accepted = expected == 0
                              ? 0
                              : output.write(
                                    reinterpret_cast<const uint8_t*>(ackLine),
                                    expected);
  // Benchmark ACKs are not product traffic.  Completing this bounded HWCDC
  // flush prevents a rendered ACK from remaining stranded until a later host
  // OUT packet wakes the ESP32-C3 USB IN endpoint.
  output.flush();
  ackBytesAccepted_ += accepted;
  if (accepted != expected) ++ackShortWrites_;
  if (innerType == 'D') emitBenchmarkDiagnostics(output, trialId);
}

void HUDProtocolParser::emitBenchmarkDiagnostics(Print& output,
                                                  const String& trialId) {
  output.print("SGDIAG|");
  output.print(trialId);
  output.print('|'); output.print(rxBytes_);
  output.print('|'); output.print(completedFrames_);
  output.print('|'); output.print(validFrames_);
  output.print('|'); output.print(crcFailures_);
  output.print('|'); output.print(parserRecoveries_);
  output.print('|'); output.print(malformedBenchmarkFrames_);
  output.print('|'); output.print(ackAttempts_);
  output.print('|'); output.print(ackBytesAccepted_);
  output.print('|'); output.print(ackShortWrites_);
  output.print('|'); output.print(output.availableForWrite());
  output.print('|'); output.print(minTxAvailable_);
  output.print('|'); output.print(pollCount_);
  output.print('|'); output.print(millis());
  output.print('|'); output.print(ESP.getFreeHeap());
  output.print('|'); output.print(ESP.getMinFreeHeap());
  output.print('|'); output.print(ESP.getMaxAllocHeap());
  output.print('|'); output.print(uxTaskGetStackHighWaterMark(nullptr));
  output.print('|'); output.print(static_cast<int>(esp_reset_reason()));
  output.print('|'); output.print(hud_.benchmarkDisplayCommitCount());
  output.print('|'); output.print(hud_.benchmarkLastDisplayMicros());
  output.print('|'); output.print(hud_.benchmarkMaxDisplayMicros());
  output.print('|'); output.print(lastParsedTrialId_);
  output.print('|'); output.println(lastAckAttemptTrialId_);
}

void HUDProtocolParser::applyFrame() {
  if (type_ == 'B') {
    if (benchmarkOutput_ != nullptr) applyBenchmarkFrame(*benchmarkOutput_);
    return;
  }
  if (type_ == 'S') hud_.setSubtitle(payload_);
  else if (type_ == 'P') hud_.setPartialSubtitle(payload_);
  else if (type_ == 'A') hud_.setAlert(payload_);
  else if (type_ == 'E') hud_.setAlertState(payload_);
  else if (type_ == 'C') hud_.setEnvironmentalSound(payload_);
  else if (type_ == 'T') hud_.setStatus(payload_);
  else if (type_ == 'K') hud_.synchronizeClock(payload_);
}

void HUDProtocolParser::consume(uint8_t value) {
  switch (state_) {
    case State::MagicS:
      if (value == 'S') state_ = State::MagicG;
      break;
    case State::MagicG:
      state_ = value == 'G' ? State::Version
                            : (value == 'S' ? State::MagicG : State::MagicS);
      break;
    case State::Version:
      if (value != 1) { reset(value, true); break; }
      crc_ = updateCrc(crc_, value);
      state_ = State::Type;
      break;
    case State::Type:
      if (value != 'S' && value != 'P' && value != 'A' && value != 'E' &&
          value != 'C' && value != 'T' && value != 'K' && value != 'B') {
        reset(value, true);
        break;
      }
      type_ = value;
      crc_ = updateCrc(crc_, value);
      state_ = State::LengthHigh;
      break;
    case State::LengthHigh:
      length_ = static_cast<uint16_t>(value) << 8;
      crc_ = updateCrc(crc_, value);
      state_ = State::LengthLow;
      break;
    case State::LengthLow:
      length_ |= value;
      crc_ = updateCrc(crc_, value);
      if (length_ > kMaxPayload) { reset(value, true); break; }
      payload_.reserve(length_ + 1);
      state_ = length_ == 0 ? State::CrcHigh : State::Payload;
      break;
    case State::Payload:
      payload_ += static_cast<char>(value);
      crc_ = updateCrc(crc_, value);
      if (++received_ == length_) state_ = State::CrcHigh;
      break;
    case State::CrcHigh:
      receivedCrc_ = static_cast<uint16_t>(value) << 8;
      state_ = State::CrcLow;
      break;
    case State::CrcLow:
      receivedCrc_ |= value;
      ++completedFrames_;
      const bool validCrc = receivedCrc_ == crc_;
      if (validCrc) ++validFrames_;
      if (receivedCrc_ == crc_) applyFrame();
      if (validCrc) {
        reset(value);
      } else {
        ++crcFailures_;
        reset(value, true);
      }
      break;
  }
}

void HUDProtocolParser::poll(Stream& input, Print* benchmarkOutput) {
  ++pollCount_;
  benchmarkOutput_ = benchmarkOutput;
  while (input.available() > 0) {
    ++rxBytes_;
    consume(static_cast<uint8_t>(input.read()));
  }
  benchmarkOutput_ = nullptr;
}
