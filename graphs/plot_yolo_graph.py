import pandas as pd
import matplotlib.pyplot as plt

filled_data = pd.read_csv('model/Final_YOLOV8_Model_bestone/train2/results.csv')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(filled_data['                  epoch'], filled_data['         train/box_loss'], label='train/box_loss')
plt.xlabel('Epoch')
plt.ylabel('Train Box Loss')
plt.title('Train Box Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(filled_data['                  epoch'], filled_data['       metrics/mAP50(B)'], label='metrics/mAP50(B)', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Metrics mAP50(B)')
plt.title('Metrics mAP50(B) over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
