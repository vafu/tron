#include <Adafruit_VL53L5CX.h>
#include <Wire.h>

namespace {

constexpr uint32_t kBaud = 115200;
constexpr uint8_t kLpnPin = 17;
constexpr uint8_t kIntPin = 16;
constexpr uint8_t kResolution = 64;
constexpr uint8_t kFrequencyHz = 15;
constexpr uint32_t kI2cClockHz = 400000;
constexpr uint16_t kInvalidDistance = 0;
constexpr uint32_t kPollFallbackUs = 5000;

Adafruit_VL53L5CX sensor;
VL53L5CX_ResultsData results;
uint32_t sequence = 0;
volatile bool interrupt_ready = false;
volatile uint32_t interrupt_ready_us = 0;
uint32_t last_poll_us = 0;

void on_tof_interrupt() {
  interrupt_ready_us = micros();
  interrupt_ready = true;
}

void halt(const __FlashStringHelper *message) {
  Serial.println(message);
  while (true) {
    delay(100);
  }
}

bool is_valid_zone(uint8_t index) {
  return results.target_status[index] == 5 && results.distance_mm[index] > 0;
}

void print_zones(uint8_t resolution) {
  for (uint8_t i = 0; i < resolution; i++) {
    if (i > 0) {
      Serial.print(',');
    }
    if (is_valid_zone(i)) {
      Serial.print(results.distance_mm[i]);
    } else {
      Serial.print(kInvalidDistance);
    }
  }
}

void print_sample(uint8_t resolution, uint32_t ready_us) {
  uint16_t min_mm = UINT16_MAX;
  uint16_t max_mm = 0;
  uint32_t sum_mm = 0;
  uint8_t valid = 0;

  for (uint8_t i = 0; i < resolution; i++) {
    if (!is_valid_zone(i)) {
      continue;
    }
    const uint16_t distance = results.distance_mm[i];
    min_mm = min(min_mm, distance);
    max_mm = max(max_mm, distance);
    sum_mm += distance;
    valid++;
  }

  const uint16_t center_mm = valid > 0 ? sum_mm / valid : 0;
  if (valid == 0) {
    min_mm = 0;
  }

  Serial.print(F("tof seq="));
  Serial.print(sequence++);
  Serial.print(F(" t_ms="));
  Serial.print(millis());
  Serial.print(F(" irq_us="));
  Serial.print(ready_us);
  Serial.print(F(" res="));
  Serial.print(resolution);
  Serial.print(F(" center_mm="));
  Serial.print(center_mm);
  Serial.print(F(" min_mm="));
  Serial.print(min_mm);
  Serial.print(F(" max_mm="));
  Serial.print(max_mm);
  Serial.print(F(" valid="));
  Serial.print(valid);
  Serial.print(F(" zones="));
  print_zones(resolution);
  Serial.println();
}

} // namespace

void setup() {
  pinMode(kLpnPin, OUTPUT);
  pinMode(kIntPin, INPUT_PULLUP);
  digitalWrite(kLpnPin, LOW);
  delay(10);
  digitalWrite(kLpnPin, HIGH);
  delay(100);

  Serial.begin(kBaud);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  Wire.begin();
  Wire.setClock(kI2cClockHz);

  Serial.println(F("tof status=boot sensor=VL53L5CX"));
  Serial.println(F("tof status=init"));

  if (!sensor.begin(VL53L5CX_DEFAULT_ADDRESS, &Wire, kI2cClockHz)) {
    halt(F("tof status=error stage=begin"));
  }

  if (!sensor.setResolution(kResolution)) {
    halt(F("tof status=error stage=set_resolution"));
  }

  if (!sensor.setRangingFrequency(kFrequencyHz)) {
    halt(F("tof status=error stage=set_frequency"));
  }

  if (!sensor.startRanging()) {
    halt(F("tof status=error stage=start_ranging"));
  }

  attachInterrupt(digitalPinToInterrupt(kIntPin), on_tof_interrupt, FALLING);

  Serial.print(F("tof status=ready res="));
  Serial.print(sensor.getResolution());
  Serial.print(F(" hz="));
  Serial.print(sensor.getRangingFrequency());
  Serial.print(F(" int_pin="));
  Serial.println(kIntPin);
}

void loop() {
  uint32_t ready_us = 0;
  bool should_check = false;

  noInterrupts();
  if (interrupt_ready) {
    ready_us = interrupt_ready_us;
    interrupt_ready = false;
    should_check = true;
  }
  interrupts();

  const uint32_t now_us = micros();
  if (!should_check && now_us - last_poll_us >= kPollFallbackUs) {
    last_poll_us = now_us;
    should_check = true;
  }

  if (should_check && sensor.isDataReady() && sensor.getRangingData(&results)) {
    if (ready_us == 0) {
      ready_us = micros();
    }
    print_sample(sensor.getResolution(), ready_us);
  }
}
