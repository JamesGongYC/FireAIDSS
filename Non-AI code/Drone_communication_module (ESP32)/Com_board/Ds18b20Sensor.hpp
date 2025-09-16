#ifndef DS18B20SENSOR_HPP
#define DS18B20SENSOR_HPP
#include "Arduino.h"
#include <OneWire.h>

class DS18B20Sensor
{
public:
    DS18B20Sensor(uint8_t dataPin);

    bool begin();
    float readTemperature();

private:
    uint8_t _dataPin;
    OneWire _oneWire;
    byte _addr[8];
};


DS18B20Sensor::DS18B20Sensor(uint8_t dataPin) : _dataPin(dataPin)
{
    _oneWire = OneWire(_dataPin);
}

bool DS18B20Sensor::begin()
{
    pinMode(_dataPin, INPUT_PULLUP);
    return true; // 这里可以根据需要添加更复杂的初始化代码
}

float DS18B20Sensor::readTemperature()
{
    byte data[12];

    if (!_oneWire.search(_addr))
    {
        _oneWire.reset_search();
        return -1000.0; // 返回一个无效的温度值，表示没有找到设备
    }

    if (OneWire::crc8(_addr, 7) != _addr[7])
    {
        return -1000.0; // CRC校验失败
    }

    _oneWire.reset();
    _oneWire.select(_addr);
    _oneWire.write(0x44, 1); // 发送温度转换命令
    delay(2);                // 等待温度转换完成

    _oneWire.reset();
    _oneWire.select(_addr);
    _oneWire.write(0xBE); // 发送读取温度寄存器命令

    for (int i = 0; i < 9; i++)
    {
        data[i] = _oneWire.read();
    }

    int16_t raw = (data[1] << 8) | data[0];
    return (float)raw / 16.0;
}

#endif // DS18B20SENSOR_HPP
 // DS18B20SENSOR_HPP
