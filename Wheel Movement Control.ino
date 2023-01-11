#define pwm1 5
#define pwm2 6
#define pwm3 9
#define pwm4 10

void setup() {
  pinMode(pwm1, OUTPUT);
  pinMode(pwm2, OUTPUT);
  pinMode(pwm3, OUTPUT);
  pinMode(pwm4, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(1);
}
//speed[0] = kiri, sebaliknya
int speed[2] = {0, 0};
int maxi = 50;
int maxa = 0.85*maxi;
int state = 1;

// Maju 
void maju(){
  speed[0]+=10;
  speed[1]+=10;
  speed[0] = speed[0] > maxi ? maxi : speed[0];
  speed[1] = speed[1] > maxa ? maxa : speed[1];
  return;
}

// Stop 
void stop(){
  speed[0]-=10;
  speed[1]-=10;
  speed[0] = speed[0] < 0 ? 0 : speed[0];
  speed[1] = speed[1] < 0 ? 0 : speed[1];
  return;
}

//Kanan
void kanan(){
  speed[0] += 10;
  speed[1] = 0;
  speed[0] = speed[0] > 35 ? 35 : speed[0];
  speed[1] = speed[1] > 0 ? 0 : speed[1];
  return;
  return;
} 

//Kiri
void kiri(){
  speed[0] = 0;
  speed[1] += 10;
  speed[0] = speed[0] > 0 ? 0 : speed[0];
  speed[1] = speed[1] > 35 ? 35 : speed[1];
  return;
} 

void loop() {
  if (Serial.available() > 0)
  {
    state = Serial.readString().toInt();  
  }
  if ((state == 2 || state == 3) && speed[0] != 0 && speed[1] != 0){
    state = 1;
  }
  switch(state){
    case 0 : maju(); break;
    case 1 : stop(); break;
    case 2 : kanan(); break;
    case 3 : kiri(); break;
  }

  analogWrite(pwm1, 0);
  analogWrite(pwm2, speed[0]);
  analogWrite(pwm3, 0);
  analogWrite(pwm4, speed[1]);
  delay(100);
  Serial.println(String(speed[0]) + "+" + String(speed[1]));
}
