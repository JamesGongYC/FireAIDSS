/*
 * FireAIDSS Data Central Module
 * ============================
 * 
 * ESP32S3-based central communication hub for FireAIDSS drone swarm system.
 * Handles bidirectional communication between laptop control station and drone swarm.
 * 
 * Hardware: ESP32S3 Dev Module
 * MAC Address: 24:58:7C:DA:22:44
 * 
 * Communication Protocol:
 * - Serial (USB): Commands from laptop, sensor data to laptop
 * - ESP-NOW: Wireless communication with drone modules
 * 
 * Command Format (from laptop): "drone_id,val1,val2,val3,val4,val5/"
 * Report Format (to laptop): "drone_id temperature wind_x wind_y wind_z"
 * 
 * Updated for FireAIDSS integration with improved error handling and status reporting.
 */

#include <esp_now.h>
#include <WiFi.h>

// Drone MAC addresses - Update these based on actual hardware
uint8_t broadcastAddress1[] = { 0x4C, 0x11, 0xAE, 0x64, 0xFF, 0x30 };  // Drone 1
uint8_t broadcastAddress2[] = { 0xE4, 0x65, 0xB8, 0x75, 0x42, 0x74 };  // Drone 2
uint8_t broadcastAddress3[] = { 0xE4, 0x65, 0xB8, 0x74, 0x4A, 0xC8 };  // Drone 3 (optional)
uint8_t broadcastAddress4[] = { 0x4C, 0x11, 0xAE, 0x65, 0xA6, 0x84 };  // Drone 4 (optional)

// ESP-NOW peer information
esp_now_peer_info_t peerInfo1, peerInfo2, peerInfo3, peerInfo4;

// Data structures
struct command_packet {
  float val1, val2, val3, val4, val5;  // Flight control values
};

struct sensor_report {
  int id;           // Drone ID
  float tmp;        // Temperature (Celsius)
  float Xspd;       // Wind X velocity (m/s)
  float Yspd;       // Wind Y velocity (m/s)
  float Zspd;       // Wind Z velocity (m/s)
};

// Global variables
sensor_report latest_reports[5];  // Store latest reports from each drone (index 0 unused)
command_packet outgoing_command;
String serial_packet = "";
String temp_string = "";
float command_params[10];
unsigned long last_report_time = 0;
unsigned long last_heartbeat = 0;
unsigned long last_command_time = 0;  // 10 Hz command transmission timing
bool drone_connected[5] = {false, false, false, false, false};  // Connection status

// State-based command queue (10 Hz transmission)
struct queued_command {
  int drone_id;
  command_packet cmd;
  bool valid;
};
queued_command command_queue[5];  // Command queue for each drone (index 0 unused)

// Status LED
const int STATUS_LED = 2;
bool led_state = false;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("FireAIDSS Data Central Module Starting...");
  
  // Initialize status LED
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);
  
  // Initialize WiFi in station mode
  WiFi.mode(WIFI_STA);
  Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());
  
  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ERROR: ESP-NOW initialization failed");
    return;
  }
  
  Serial.println("ESP-NOW initialized successfully");
  
  // Register callbacks
  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataReceived);
  
  // Add peers (drone communication modules)
  addPeer(&peerInfo1, broadcastAddress1, "Drone 1");
  addPeer(&peerInfo2, broadcastAddress2, "Drone 2");
  // Uncomment for additional drones:
  // addPeer(&peerInfo3, broadcastAddress3, "Drone 3");
  // addPeer(&peerInfo4, broadcastAddress4, "Drone 4");
  
  // Initialize sensor report array
  for (int i = 1; i <= 4; i++) {
    latest_reports[i].id = i;
    latest_reports[i].tmp = 0.0;
    latest_reports[i].Xspd = 0.0;
    latest_reports[i].Yspd = 0.0;
    latest_reports[i].Zspd = 0.0;
  }
  
  // Initialize command queue
  for (int i = 1; i <= 4; i++) {
    command_queue[i].drone_id = i;
    command_queue[i].cmd.val1 = 1500;  // Neutral PWM values
    command_queue[i].cmd.val2 = 1500;
    command_queue[i].cmd.val3 = 1500;
    command_queue[i].cmd.val4 = 1500;
    command_queue[i].cmd.val5 = 0;
    command_queue[i].valid = false;
  }
  
  Serial.println("FireAIDSS Data Central Module Ready");
  Serial.println("Waiting for commands from control station...");
  
  // Initial status indication
  for (int i = 0; i < 5; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(200);
    digitalWrite(STATUS_LED, LOW);
    delay(200);
  }
}

void loop() {
  // Handle incoming serial commands from laptop
  handleSerialInput();
  
  // Send queued commands at 10 Hz
  sendQueuedCommands();
  
  // Send periodic sensor reports to laptop
  sendSensorReports();
  
  // Send periodic heartbeat
  sendHeartbeat();
  
  // Update status LED
  updateStatusLED();
  
  delay(10);  // Small delay for stability
}

void handleSerialInput() {
  if (Serial.available()) {
    delay(2);  // Allow complete packet to arrive
    
    // Read complete packet
    while (Serial.available()) {
      serial_packet += char(Serial.read());
    }
    
    // Parse and execute command
    if (serial_packet.length() > 0) {
      parseCommand(serial_packet);
      serial_packet = "";  // Clear for next packet
    }
  }
}

void parseCommand(String packet) {
  temp_string = "";
  int param_index = 0;
  
  // Parse comma-separated values ending with '/'
  for (int i = 0; i < packet.length(); i++) {
    char c = packet[i];
    
    if (c == ',') {
      // Store parameter
      if (param_index < 10) {
        command_params[param_index] = temp_string.toFloat();
        param_index++;
      }
      temp_string = "";
    }
    else if (c == '/') {
      // End of packet - store final parameter
      if (param_index < 10) {
        command_params[param_index] = temp_string.toFloat();
      }
      
      // Execute command
      executeCommand();
      return;
    }
    else {
      temp_string += c;
    }
  }
}

void executeCommand() {
  int drone_id = (int)command_params[0];
  
  // Validate drone ID
  if (drone_id < 1 || drone_id > 4) {
    Serial.println("ERROR: Invalid drone ID: " + String(drone_id));
    return;
  }
  
  // Store command in queue for 10 Hz transmission
  command_queue[drone_id].drone_id = drone_id;
  command_queue[drone_id].cmd.val1 = command_params[1];
  command_queue[drone_id].cmd.val2 = command_params[2];
  command_queue[drone_id].cmd.val3 = command_params[3];
  command_queue[drone_id].cmd.val4 = command_params[4];
  command_queue[drone_id].cmd.val5 = command_params[5];
  command_queue[drone_id].valid = true;
  
  // Report command queued
  Serial.println("CMD_QUEUED: Drone " + String(drone_id) + " -> " + 
                 String(command_queue[drone_id].cmd.val1) + "," + 
                 String(command_queue[drone_id].cmd.val2) + "," +
                 String(command_queue[drone_id].cmd.val3) + "," + 
                 String(command_queue[drone_id].cmd.val4) + "," +
                 String(command_queue[drone_id].cmd.val5));
}

void sendQueuedCommands() {
  // Send queued commands at 10 Hz (100ms interval)
  if (millis() - last_command_time > 100) {
    for (int i = 1; i <= 4; i++) {
      if (command_queue[i].valid) {
        esp_err_t result = ESP_FAIL;
        
        // Send command to appropriate drone
        switch (i) {
          case 1:
            result = esp_now_send(peerInfo1.peer_addr, (uint8_t*)&command_queue[i].cmd, sizeof(command_queue[i].cmd));
            break;
          case 2:
            result = esp_now_send(peerInfo2.peer_addr, (uint8_t*)&command_queue[i].cmd, sizeof(command_queue[i].cmd));
            break;
          case 3:
            result = esp_now_send(peerInfo3.peer_addr, (uint8_t*)&command_queue[i].cmd, sizeof(command_queue[i].cmd));
            break;
          case 4:
            result = esp_now_send(peerInfo4.peer_addr, (uint8_t*)&command_queue[i].cmd, sizeof(command_queue[i].cmd));
            break;
        }
        
        // Report transmission status
        if (result == ESP_OK) {
          Serial.println("CMD_SENT: Drone " + String(i) + " @ 10Hz -> " + 
                         String(command_queue[i].cmd.val1) + "," + String(command_queue[i].cmd.val2) + "," +
                         String(command_queue[i].cmd.val3) + "," + String(command_queue[i].cmd.val4) + "," +
                         String(command_queue[i].cmd.val5));
        } else {
          Serial.println("CMD_FAIL: Drone " + String(i) + " transmission failed");
        }
        
        // Keep command valid for continuous transmission at 10 Hz
        // Commands remain active until overwritten by new commands
      }
    }
    last_command_time = millis();
  }
}

void sendSensorReports() {
  // Send sensor data every 500ms
  if (millis() - last_report_time > 50) {
    for (int i = 1; i <= 2; i++) {  // Only drones 1 and 2 for now
      if (drone_connected[i]) {
        // Format: "drone_id temperature wind_x wind_y wind_z"
        String report = String(latest_reports[i].id) + " " + 
                       String(latest_reports[i].tmp, 2) + " " +
                       String(latest_reports[i].Xspd, 2) + " " +
                       String(latest_reports[i].Yspd, 2) + " " +
                       String(latest_reports[i].Zspd, 2);
        
        Serial.println(report);
        delay(10);  // Small delay between reports
      }
    }
    last_report_time = millis();
  }
}

void sendHeartbeat() {
  // Send system status every 5 seconds
  if (millis() - last_heartbeat > 5000) {
    String status = "STATUS: ";
    for (int i = 1; i <= 2; i++) {
      status += "D" + String(i) + ":" + (drone_connected[i] ? "OK" : "DISC") + " ";
    }
    Serial.println(status);
    last_heartbeat = millis();
  }
}

void updateStatusLED() {
  // Blink LED to indicate system activity
  static unsigned long last_blink = 0;
  if (millis() - last_blink > 1000) {
    led_state = !led_state;
    digitalWrite(STATUS_LED, led_state);
    last_blink = millis();
  }
}

// ESP-NOW callback functions
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Optional: Handle send confirmation
  if (status != ESP_NOW_SEND_SUCCESS) {
    Serial.println("SEND_FAIL: Command delivery failed");
  }
}

void onDataReceived(const uint8_t *mac, const uint8_t *incoming_data, int len) {
  // Receive sensor reports from drones
  sensor_report received_report;
  memcpy(&received_report, incoming_data, sizeof(received_report));
  
  // Validate drone ID
  if (received_report.id >= 1 && received_report.id <= 4) {
    // Update latest report
    latest_reports[received_report.id] = received_report;
    
    // Update connection status
    drone_connected[received_report.id] = true;
    
    // Optional: Debug output
    /*
    Serial.println("SENSOR: Drone " + String(received_report.id) + 
                   " T=" + String(received_report.tmp, 1) + "°C " +
                   "W=[" + String(received_report.Xspd, 1) + "," + 
                   String(received_report.Yspd, 1) + "," + 
                   String(received_report.Zspd, 1) + "] m/s");
    */
  }
}

// Helper function to add ESP-NOW peers
void addPeer(esp_now_peer_info_t *peer, uint8_t *mac_addr, String drone_name) {
  peer->channel = 0;
  peer->encrypt = false;
  memcpy(peer->peer_addr, mac_addr, 6);
  
  if (esp_now_add_peer(peer) != ESP_OK) {
    Serial.println("ERROR: Failed to add peer - " + drone_name);
  } else {
    Serial.println("SUCCESS: Added peer - " + drone_name);
  }
}
