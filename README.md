# Data Analysis and Preprocessing

## Top Section with Setup Commands

### Ensure necessary libraries are installed
!pip install pandas matplotlib seaborn scikit-learn

### Download the dataset
!wget -O data.csv "link_to_your_data.csv"

### Unzip if necessary
### !unzip data.zip -d data_folder

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the data
data = pd.read_csv("data.csv")

# Display the data
data.head()
