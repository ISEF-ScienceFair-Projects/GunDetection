#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
#include <string.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
void setup() {
  // put your setup code here, to run once
Serial.begin(9600);
pinMode(13,OUTPUT);
  // initialize the LCD
  lcd.begin();

  // Turn on the blacklight and print a message.
  lcd.backlight();
  

}

void loop() {
 /*
lcd.clear();
lcd.print(String(Serial.read()));
delay(100);
Serial.print("lol");
delay(1000);
 */

if (String(Serial.read()) == "48"){
    Serial.write('5');
    lcd.clear();
    lcd.print("Gunman: 0");
    }
  
 if (String(Serial.read()) == "49"){
    Serial.write('5');
    lcd.clear();
    lcd.print("Gunman: 1");
    }
   
 if (String(Serial.read()) == "50"){
    Serial.write('5');
    lcd.clear();
    lcd.print("Gunman: 2");
    }
 if (String(Serial.read()) == "51"){
    Serial.write('5');
    lcd.clear();
    lcd.print("Gunman: 3");
    }

}
