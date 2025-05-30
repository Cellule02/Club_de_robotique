int trigger_pin1 = 2;
int echo_pin1 = 3;
int trigger_pin2 = 7;
int echo_pin2 = 6;
int trigger_pin3 = 13;
int echo_pin3 = 12;

// trigger stop du robot
int stop_pin=4;

long distance1, pulse_duration1;
long distance2, pulse_duration2;
long distance3, pulse_duration3;
long min_dist;

void setup() {
  Serial.begin (9600);

  pinMode(trigger_pin1, OUTPUT);
  pinMode(echo_pin1, INPUT);

  pinMode(trigger_pin2, OUTPUT);
  pinMode(echo_pin2, INPUT);

  pinMode(trigger_pin3, OUTPUT);
  pinMode(echo_pin3, INPUT);

  pinMode(stop_pin, OUTPUT);

  digitalWrite(trigger_pin1, LOW);
  digitalWrite(trigger_pin2, LOW);
  digitalWrite(trigger_pin3, LOW);

  digitalWrite(stop_pin, HIGH);
}

void loop() {
  digitalWrite(trigger_pin1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigger_pin1, LOW);

  pulse_duration1 = pulseIn(echo_pin1, HIGH);

  distance1 = round(pulse_duration1 * 0.0171);


  digitalWrite(trigger_pin2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigger_pin2, LOW);

  pulse_duration2 = pulseIn(echo_pin2, HIGH);

  distance2 = round(pulse_duration2 * 0.0171);

  digitalWrite(trigger_pin3, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigger_pin3, LOW);

  pulse_duration3 = pulseIn(echo_pin3, HIGH);

  distance3 = round(pulse_duration3 * 0.0171);


  if (distance1 < 60){
    digitalWrite(stop_pin, LOW);
  }
  else if (distance2 < 60){
    digitalWrite(stop_pin, LOW);
  }
  else if (distance3 < 60){
    digitalWrite(stop_pin, LOW);
  }
  else {
    digitalWrite(stop_pin, HIGH);
  }
  
  /*//distance = round(pulse_duration/0.00675);
  Serial.print("cap1  ");
  Serial.print(distance1);
  Serial.print("cm");
  Serial.println();

  Serial.print("cap2  ");
  Serial.print(distance2);
  Serial.print("cm");
  Serial.println();

  Serial.print("cap3  ");
  Serial.print(distance3);
  Serial.print("cm");
  Serial.println();*/

  Serial.println();
  Serial.print("distance_minimal: ");
  Serial.print(min_dist);
  Serial.println();

  delay(1000);
}