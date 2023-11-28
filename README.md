# Science Fair 2024: 
## Identifying Guns using Machine Learning with a Rapid Response System to Save Lives
<p style="text-align: center;">
In 2021, 48,830 people died from gun-related injuries in the U.S., according to the CDC. When there is an active gunman in the building, most people don't realize it until it is too late. In most of these scenarios, a rapid response from law enforcement will significantly reduce the number of victims. By utilizing Machine Learning, we can design and build an autonomous rapid-response system. By using convolutional neural networks, this system will identify a person with a gun and immediately notify the police with a picture of the person's description and precise location. This will reduce the time in which the police would arrive at the scene and therefore save more lives. This system will continue to track the person carrying the gun to provide the police with real-time location data. This solution can be applied to save lives in mass shooting scenarios in buildings such as Malls, Supermarkets, and schools.
</p>

## Files

`https://drive.google.com/drive/folders/1K7W09SLk2A7Yeg40mQ8TqBjQ4C44ZNG8?usp=drive_link`

Download Gun model weights and Darknet clothes detection weights

## Password for Texting
`https://drive.google.com/drive/folders/10lVkmILzi5nVou20RG_yyEVbyhk1iEXK`
download this file and put it in the same directory as main.py in the gateway texting folder

## Usage

download the weights and paste the `yolo-obj_last.weights` file into the main directory, and paste `built_model` folder into the clothes directory.
<br>
then run 

```shell
python3 gun.py
```

for the gun AI
<br>
and run

```shell
python3 clothes.py
```
for the clothes detection

## Credits
https://github.com/Rabbit1010/Clothes-Recognition-and-Retrieval -
For the cloth detecion code
<br>
The rest is our own.