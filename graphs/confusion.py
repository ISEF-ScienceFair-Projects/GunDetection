import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#plug in data
conf_matrix = np.array([[79, 7],
                        [1, 53]])

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Gun", "No Gun"],
            yticklabels=["Gun", "No Gun"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Final Model")

plt.show()

