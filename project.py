import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
df=pd.read_csv(r"C:\Users\Hp\Downloads\Pollution.csv")
print(df.info())
print(df.shape)
print(df.head())
print(df.tail())
print("Checking Null values: ", df.isnull())
print("Total Null values: ", df.isnull().sum().sum())
df1=df.dropna(subset=['pollutant_avg'])

#Visualiztion
#Top States using Bar chart
plt.figure(figsize=(12,6))
state_avg=df1.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=state_avg.values, y=state_avg.index,hue=state_avg.index,legend=False, palette='viridis')
plt.title("Top 15 states by Average Pollutant Level")
plt.show()

#Min vs Max Pollutants Level
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df1, x='pollutant_min', y='pollutant_max', hue='pollutant_id', alpha=0.7)
plt.title('Scatter Plot: Pollutant Min vs Max Levels')
plt.xlabel('Minimum Pollutant Level')
plt.ylabel('Maximum Pollutant Level')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Pollutant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Distribution of Pollutant types
plt.figure(figsize=(8, 8))
po = df1['pollutant_id'].value_counts()
plt.pie(po, labels=po.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Percentage Distribution of Pollutant Types')
plt.show()


# Pollutant Distribution using Bocplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df1, x='pollutant_id' ,y='pollutant_avg')
plt.title('Pollutant Level Distribution by Type')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8,6))
corr=df1[['latitude','longitude','pollutant_min','pollutant_max','pollutant_avg']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt= ".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

#Monitoring Stations Per State
plt.figure(figsize=(12, 6))
st_c= df.groupby('state')['station'].nunique().sort_values(ascending=False)
sns.barplot(x=st_c.values, y=st_c.index,hue=st_c.index,legend=False, palette='magma')
plt.title('Number of Unique Monitoring Stations per State')
plt.xlabel('Count of Stations')
plt.show()

#Frequency of Pollutant Types
plt.figure(figsize=(10, 6))
pc = df['pollutant_id'].value_counts()
sns.barplot(x=pc.index, y=pc.values,hue=pc.values,legend=False, palette='plasma')
plt.title('Frequency of Pollutant Types in Dataset')
plt.show()

#Average PM2.5 Levels per State
pm25 = df1[df1['pollutant_id'] == 'PM2.5']
plt.figure(figsize=(12, 6))
p25_avg = pm25.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False)
sns.barplot(x=p25_avg.values, y=p25_avg.index,hue=p25_avg.index,legend=False, palette='Reds_r')
plt.title('Average $PM_{2.5}$ Levels per State')
plt.show()


#ML models

# Encode Categorical Data
le_state = LabelEncoder()
df1['state_encoded'] = le_state.fit_transform(df1['state'])
le_pollutant = LabelEncoder()
df1['pollutant_encoded'] = le_pollutant.fit_transform(df1['pollutant_id'])

# Define Features and Target
x = df1[['state_encoded', 'pollutant_encoded', 'latitude', 'longitude']]
y = df1['pollutant_avg']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")




