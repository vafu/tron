# tof-teensy-arduino

Arduino/Teensy bridge firmware for the VL53L5CX TOF sensor.

Hardware wiring:

- Teensy 4.0 pin 19: I2C SCL
- Teensy 4.0 pin 18: I2C SDA
- Teensy 4.0 pin 17: sensor `LPn`, driven high during startup
- Teensy 4.0 pin 16: sensor `INT`, active-low range-ready interrupt

The sketch streams one serial line per range frame at `115200` baud:

```text
tof seq=<n> t_ms=<millis> irq_us=<micros> res=<16|64> center_mm=<mm> min_mm=<mm> max_mm=<mm> valid=<n> zones=<csv>
```

Useful commands:

```sh
arduino-cli compile --fqbn teensy:avr:teensy40 tools/tof-teensy-arduino/tof_bridge
arduino-cli upload --fqbn teensy:avr:teensy40 tools/tof-teensy-arduino/tof_bridge
```
