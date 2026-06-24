#include <Servo.h>
#define trig 7
#define echo 6

Servo servol;

int luz = 3;
int servo_motor = 2;
int cm;
int duration;

void setup(){
  //LED
  pinMode(luz, OUTPUT);
  
  //Servo motor
  servol.attach(servo_motor);
  
  //Sensor
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
}


void loop(){
  digitalWrite(trig, LOW);
  delayMicroseconds(5);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  
  duration = pulseIn(echo, HIGH);
  cm = (duration/2) / 29.1;

  
  if(cm < 55){
    digitalWrite(luz, HIGH); //LED
    servol.write(0); //Motor
  }
  
  else{
   digitalWrite(luz, LOW); //LED 
   servol.write(180); //Motor
  }
  
}
