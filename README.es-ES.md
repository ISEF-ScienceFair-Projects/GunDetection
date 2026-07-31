

# Feria de Ciencias 2024: 
## Identificación de Armas de Fuego mediante Aprendizaje Automático con un Sistema de Respuesta Rápida para Salvar Vidas
<p style="text-align: center;">
En 2021, 48.830 personas murieron por lesiones relacionadas con armas de fuego en EE. UU., según los CDC. Cuando hay un tirador activo en un edificio, la mayoría de las personas no se dan cuenta hasta que es demasiado tarde. En la mayoría de estos escenarios, una respuesta rápida de las fuerzas del orden reducirá significativamente el número de víctimas. Mediante el uso del Aprendizaje Automático, podemos diseñar y construir un sistema autónomo de respuesta rápida. Utilizando redes neuronales convolucionales, este sistema identificará a una persona con un arma y notificará inmediatamente a la policía con una imagen que incluya la descripción de la persona y su ubicación precisa. Esto reducirá el tiempo de llegada de la policía a la escena y, por lo tanto, salvará más vidas. Este sistema continuará rastreando a la persona que porta el arma para proporcionar a la policía datos de ubicación en tiempo real. Esta solución puede aplicarse para salvar vidas en escenarios de tiroteos masivos en edificios como centros comerciales, supermercados y escuelas.
</p>

## Archivos

`https://drive.google.com/drive/folders/1K7W09SLk2A7Yeg40mQ8TqBjQ4C44ZNG8?usp=drive_link`

Descargar los pesos del modelo de armas y los pesos de detección de ropa de Darknet

## Contraseña para los Mensajes de Texto
[no está aquí]

Descargue este archivo y colóquelo en el mismo directorio que main.py en la carpeta gateway texting

## Uso

descargue los pesos y pegue el archivo `yolo-obj_last.weights` en el directorio principal, y pegue la carpeta `built_model` en el directorio de ropa.
<br>
luego ejecute 

```shell
python3 gun.py
```

para la IA de detección de armas
<br>
y ejecute

```shell
python3 clothes.py
```
para la detección de ropa

## Créditos
https://github.com/Rabbit1010/Clothes-Recognition-and-Retrieval -
Por el código de detección de ropa
<br>
El resto es nuestro.
