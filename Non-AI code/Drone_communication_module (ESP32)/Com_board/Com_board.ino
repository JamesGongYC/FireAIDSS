/*
 * FireAIDSS Drone Communication Module
 * ===================================
 * 
 * ESP32-based communication interface for individual drones in FireAIDSS system.
 * Handles wireless communication with central module and sensor data collection.
 * 
 * Hardware: ESP32 Minikit
 * 
 * IMPORTANT: Update DRONE_ID and broadcastAddress before uploading to each drone!
 * 
 * Drone 1: MAC 4C:11:AE:64:FF:30, DRONE_ID = 1
 * Drone 2: MAC E4:65:B8:75:42:74, DRONE_ID = 2
 * 
 * Communication Protocol:
 * - ESP-NOW: Receives commands from central module, sends sensor reports
 * - Serial: Forwards commands to flight controller, receives status
 * 
 * Sensors:
 * - 4x DS18B20 temperature sensors
 * - 3x AS5600 wind speed sensors via I2C multiplexer
 * 
 * Updated for FireAIDSS integration with improved sensor handling and error recovery.
 */

#if CONFIG_FREERTOS_UNICORE
#define ARDUINO_RUNNING_CORE 0
#else
#define ARDUINO_RUNNING_CORE 1
#endif

#include <esp_now.h>
#include <WiFi.h>
#include "Ds18b20Sensor.hpp"
#include "AS5600.h"
#include "TCA9548.h"

// *** IMPORTANT: UPDATE THESE FOR EACH DRONE ***
const int DRONE_ID = 1;  // Change to 1, 2, 3, or 4 for each drone
uint8_t centralModuleAddress[] = { 0x24, 0x58, 0x7C, 0xDA, 0x22, 0x44 };  // Central module MAC

// Flight control PWM values
uint16_t flight_vals[5] = {1500, 1500, 1500, 1500, 0};  // roll, pitch, throttle, yaw, mode
uint16_t takeoff_flag = 0;
unsigned long takeoff_timer = 0;
unsigned long last_command_time = 0;
unsigned long last_sensor_report = 0;

// Sensor hardware
DS18B20Sensor tempSensor1(26);
DS18B20Sensor tempSensor2(18);
DS18B20Sensor tempSensor3(19);
DS18B20Sensor tempSensor4(23);

AS5600 windSensor1;
AS5600 windSensor2;  
AS5600 windSensor3;
PCA9546 i2c_multiplexer(0x70);

// Sensor data
float temperature_readings[4] = {0.0, 0.0, 0.0, 0.0};
float wind_readings[3] = {0.0, 0.0, 0.0};
float avg_temperature = 20.0;  // Default temperature

// Communication structures
struct command_packet {
  float val1, val2, val3, val4, val5;
};

struct sensor_report {
  int id;
  float tmp;
  float Xspd;
  float Yspd;
  float Zspd;
};

command_packet received_command;
sensor_report outgoing_report;
esp_now_peer_info_t centralPeer;

// Status indicators
const int STATUS_LED = 2;
bool sensor_error = false;
bool communication_active = false;

// Task handles
void sensorTask(void *pvParameters);
void communicationTask(void *pvParameters);

void setup() {
  // Initialize serial communications
  Serial.begin(115200);  // Debug/monitoring
  Serial1.begin(115200, SERIAL_8N1, 16, 17);  // Flight controller communication
  
  pinMode(STATUS_LED, OUTPUT);
  Serial.println("FireAIDSS Drone Communication Module Starting...");
  Serial.println("Drone ID: " + String(DRONE_ID));
  
  // Initialize I2C for sensors
  Wire.begin(22, 21, 400000UL);
  pinMode(22, INPUT_PULLUP);
  pinMode(21, INPUT_PULLUP);
  
  // Initialize temperature sensors
  Serial.println("Initializing temperature sensors...");
  tempSensor1.begin();
  tempSensor2.begin();
  tempSensor3.begin();
  tempSensor4.begin();
  
  // Initialize I2C multiplexer and wind sensors
  Serial.println("Initializing wind sensors...");
  i2c_multiplexer.begin();
  
  i2c_multiplexer.selectChannel(1);
  windSensor1.begin(4);
  windSensor1.setDirection(AS5600_CLOCK_WISE);
  
  i2c_multiplexer.selectChannel(2);
  windSensor2.begin(32);
  windSensor2.setDirection(AS5600_CLOCK_WISE);
  
  i2c_multiplexer.selectChannel(0);
  windSensor3.begin(25);
  windSensor3.setDirection(AS5600_CLOCK_WISE);
  
  // Initialize WiFi and ESP-NOW
  WiFi.mode(WIFI_STA);
  Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());
  
  if (esp_now_init() != ESP_OK) {
    Serial.println("ERROR: ESP-NOW initialization failed");
    return;
  }
  
  // Register callbacks
  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataReceived);
  
  // Add central module as peer
  centralPeer.channel = 0;
  centralPeer.encrypt = false;
  memcpy(centralPeer.peer_addr, centralModuleAddress, 6);
  
  if (esp_now_add_peer(&centralPeer) != ESP_OK) {
    Serial.println("ERROR: Failed to add central module peer");
    return;
  }
  
  Serial.println("ESP-NOW initialized successfully");
  
  // Create FreeRTOS tasks
  xTaskCreatePinnedToCore(
    sensorTask,
    "Sensor Reading Task",
    4096,
    NULL,
    1,  // Priority
    NULL,
    ARDUINO_RUNNING_CORE
  );
  
  xTaskCreatePinnedToCore(
    communicationTask,
    "Communication Task",
    4096,
    NULL,
    1,  // Priority
    NULL,
    ARDUINO_RUNNING_CORE
  );
  
  // Initialize sensor report structure
  outgoing_report.id = DRONE_ID;
  
  Serial.println("Drone communication module ready");
  
  // Startup indication
  for (int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(200);
    digitalWrite(STATUS_LED, LOW);
    delay(200);
  }
}

void loop() {
  // Handle flight controller communication
  handleFlightController();
  
  // Update status LED
  updateStatusLED();
  
  delay(10);
}

void handleFlightController() {
  // Handle takeoff sequence
  if (takeoff_flag == 1 && millis() - takeoff_timer < 1000) {
    // Takeoff command active
    Serial1.print(":1500,1500,1500,1500,2#");
    Serial.println("TAKEOFF: Sending takeoff command");
  }
  else if (takeoff_flag == 1) {
    // Takeoff sequence complete
    Serial1.print(":1500,1500,1500,1500,3#");
    Serial.println("TAKEOFF: Sequence complete");
    takeoff_flag = 0;
    last_command_time = millis();
  }
  else if (millis() - last_command_time > 100 && flight_vals[4] != 1) {
    // Send regular flight commands
    String command = ":" + String(flight_vals[0]) + "," + 
                    String(flight_vals[1]) + "," + 
                    String(flight_vals[2]) + "," + 
                    String(flight_vals[3]) + "," + 
                    String(flight_vals[4]) + "#";
    
    Serial1.print(command);
    Serial.println("FC_CMD: " + command);
    last_command_time = millis();
  }
  else if (flight_vals[4] == 1) {
    // Landing command
    Serial1.print(":1500,1500,1500,1500,1#");
    Serial.println("LANDING: Emergency landing");
    flight_vals[4] = 0;
    flight_vals[2] = 1000;  // Reduce throttle
  }
}

void updateStatusLED() {
  static unsigned long last_blink = 0;
  static bool led_state = false;
  
  unsigned long blink_interval = communication_active ? 500 : 1000;
  if (sensor_error) blink_interval = 100;  // Fast blink for errors
  
  if (millis() - last_blink > blink_interval) {
    led_state = !led_state;
    digitalWrite(STATUS_LED, led_state);
    last_blink = millis();
  }
}

// FreeRTOS task for sensor reading
void sensorTask(void *pvParameters) {
  for (;;) {
    sensor_error = false;
    
    // Read temperature sensors
    for (int i = 0; i < 4; i++) {
      float temp = -1000.0;
      
      switch (i) {
        case 0: temp = tempSensor1.readTemperature(); break;
        case 1: temp = tempSensor2.readTemperature(); break;
        case 2: temp = tempSensor3.readTemperature(); break;
        case 3: temp = tempSensor4.readTemperature(); break;
      }
      
      // Use previous reading if sensor error
      if (temp != -1000.0) {
        temperature_readings[i] = temp;
      } else {
        sensor_error = true;
      }
    }
    
    // Calculate average temperature
    float temp_sum = 0;
    int valid_temps = 0;
    for (int i = 0; i < 4; i++) {
      if (temperature_readings[i] > -100) {  // Valid temperature
        temp_sum += temperature_readings[i];
        valid_temps++;
      }
    }
    
    if (valid_temps > 0) {
      avg_temperature = temp_sum / valid_temps;
    }
    
    // Read wind sensors
    i2c_multiplexer.selectChannel(1);
    wind_readings[0] = windSensor1.getAngularSpeed();
    
    i2c_multiplexer.selectChannel(2);
    wind_readings[1] = windSensor2.getAngularSpeed();
    
    i2c_multiplexer.selectChannel(0);
    wind_readings[2] = windSensor3.getAngularSpeed();
    
    // Validate wind readings
    for (int i = 0; i < 3; i++) {
      if (isnan(wind_readings[i]) || wind_readings[i] < -100 || wind_readings[i] > 100) {
        wind_readings[i] = 0.0;  // Default to zero for invalid readings
        sensor_error = true;
      }
    }
    
    vTaskDelay(50 / portTICK_PERIOD_MS);  // 20 Hz sensor reading
  }
}

// FreeRTOS task for communication
void communicationTask(void *pvParameters) {
  for (;;) {
    // Send sensor report every 100ms
    if (millis() - last_sensor_report > 100) {
      outgoing_report.tmp = avg_temperature;
      outgoing_report.Xspd = wind_readings[0];
      outgoing_report.Yspd = wind_readings[1];
      outgoing_report.Zspd = wind_readings[2];
      
      esp_err_t result = esp_now_send(0, (uint8_t*)&outgoing_report, sizeof(outgoing_report));
      
      if (result == ESP_OK) {
        communication_active = true;
        /*
        Serial.println("SENSOR_REPORT: T=" + String(avg_temperature, 1) + 
                      "°C W=[" + String(wind_readings[0], 1) + "," + 
                      String(wind_readings[1], 1) + "," + 
                      String(wind_readings[2], 1) + "]");
        */
      } else {
        communication_active = false;
        Serial.println("COMM_ERROR: Failed to send sensor report");
      }
      
      last_sensor_report = millis();
    }
    
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

// ESP-NOW callbacks
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    communication_active = false;
  }
}

void onDataReceived(const uint8_t *mac, const uint8_t *incoming_data, int len) {
  memcpy(&received_command, incoming_data, sizeof(received_command));
  
  // Update flight control values
  flight_vals[0] = (uint16_t)received_command.val1;  // Roll
  flight_vals[1] = (uint16_t)received_command.val2;  // Pitch
  flight_vals[2] = (uint16_t)received_command.val3;  // Throttle
  flight_vals[3] = (uint16_t)received_command.val4;  // Yaw
  flight_vals[4] = (uint16_t)received_command.val5;  // Mode
  
  // Handle special commands
  if (flight_vals[4] == 2) {
    // Takeoff command
    takeoff_flag = 1;
    takeoff_timer = millis();
  }
  
  communication_active = true;
  
  Serial.println("CMD_RECV: [" + String(flight_vals[0]) + "," + 
                String(flight_vals[1]) + "," + String(flight_vals[2]) + "," + 
                String(flight_vals[3]) + "," + String(flight_vals[4]) + "]");
}
