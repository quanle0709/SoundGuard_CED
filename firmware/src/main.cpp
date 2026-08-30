#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <U8g2_for_Adafruit_GFX.h>
#include <Wire.h>

#include "hud.h"
#include "protocol.h"

#ifndef HUD_OLED_ADDRESS
#define HUD_OLED_ADDRESS 0x3C
#endif
#ifndef HUD_OLED_SDA
#define HUD_OLED_SDA 21
#endif
#ifndef HUD_OLED_SCL
#define HUD_OLED_SCL 22
#endif
#ifndef HUD_OLED_WIDTH
#define HUD_OLED_WIDTH 64
#endif
#ifndef HUD_OLED_HEIGHT
#define HUD_OLED_HEIGHT 32
#endif

Adafruit_SSD1306 display(HUD_OLED_WIDTH, HUD_OLED_HEIGHT, &Wire, -1);
U8G2_FOR_ADAFRUIT_GFX unicodeText;
SoundGuardHUD hud(display, unicodeText);
HUDProtocolParser protocol(hud);

void setup() {
  Serial.begin(115200);
  Wire.begin(HUD_OLED_SDA, HUD_OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, HUD_OLED_ADDRESS)) {
    while (true) delay(1000);
  }
  hud.begin();
}

void loop() {
  protocol.poll(Serial, &Serial);
  hud.update();
  delay(1);
}
