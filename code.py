# --------------
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

# Code starts here
df = pd.read_csv(path)

df = df[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Preferred Positions', 'Wage']]


# Code ends here


# --------------
# Removes the symbol from values
import numpy as np
df['Unit'] = df['Value'].str[-1]
df['Value (M)'] = np.where(df['Unit'] == '0', 0, df['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))
df['Value (M)'] = df['Value (M)'].astype(float)
df['Value (M)'] = np.where(df['Unit'] == 'M', df['Value (M)'], df['Value (M)']/1000)

# Removes the symbol from Wage
df['Unit2'] = df['Wage'].str[-1]
df['Wage (M)'] = np.where(df['Unit2'] == '0', 0, df['Wage'].str[1:-1].replace(r'[a-zA-Z]', ''))
df['Wage (M)'] = df['Wage (M)'].astype(float)
df['Wage (M)'] = np.where(df['Unit2'] == 'M', df['Wage (M)'], df['Wage (M)']/1000)

# Drop the Unit and Unit2 from df
df = df.drop(['Unit', 'Unit2'], 1)

# New column position
df['Position'] = df['Preferred Positions'].str.split().str[0]


# --------------
import seaborn as sns
# groups of player by there position
plt.figure(figsize=(16,8))
plt.title('Grouping players by Prefered Position', fontsize=18, fontweight='bold', y=1.05,)
plt.xlabel('Number of players', fontsize=12)
plt.ylabel('Players Age', fontsize=12)
sns.countplot(x="Position", data= df)

    
    
# Wage distribution of top 100 players
value_distribution_values = df.sort_values("Wage (M)", ascending=False).reset_index().head(100)[["Name", "Wage (M)"]]
plt.figure(figsize=(16,8))
plt.title('Top 100 Players Wage Distribution', fontsize=20, fontweight='bold')
plt.ylabel('Player Wage [M€]', fontsize=15)
sns.set_style("whitegrid")
plt.plot(value_distribution_values['Wage (M)'])

    
# Comparision graph of Overall vs values(M)
overall = df.sort_values('Overall')['Overall'].unique()

    
overall_value = df.groupby(['Overall'])['Value (M)'].mean()
    
plt.figure()
plt.figure(figsize=(16,8))
plt.title('Overall vs Value', fontsize=20, fontweight='bold')
plt.xlabel('Overall', fontsize=15)
plt.ylabel('Value', fontsize=15)
sns.set_style("whitegrid")
plt.plot(overall, overall_value, label="Values in [M€]")
plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

    



# --------------

p_list_1= ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

p_list_2 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

# Code starts here
df_copy = df.copy()
storey = []
for i in p_list_1:
    storey.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df1= pd.DataFrame(np.array(storey).reshape(11, 3), columns=['Position', 'Player', 'Overall'])


# p_list_2 stats
df_copy = df.copy()
storey = []
for i in p_list_2:
    storey.append([i,
                    df_copy.loc[[df_copy[df_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(
                        index=False), df_copy[df_copy['Position'] == i]['Overall'].max()])
df_copy.drop(df_copy[df_copy['Position'] == i]['Overall'].idxmax(), inplace=True)
# return store
df2= pd.DataFrame(np.array(storey).reshape(11, 3), columns=['Position', 'Player', 'Overall'])

if df1['Overall'].mean() > df2['Overall'].mean():
        print(df1)
        print(p_list_1)
else:
    print(df2)
    print(p_list_2)
        
    
    
    



# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split


# Code starts here
X = df[['Overall', 'Potential' ,'Wage (M)']]
y = df['Value (M)']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)
r2 = r2_score(y_test,y_pred)
print("R2 Score: ",r2)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute error: ",mae)
# Code ends here


# --------------
from sklearn.preprocessing import PolynomialFeatures

# Code starts here

poly = PolynomialFeatures(degree = 3)
X_train_2  = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_2,y_train)
X_test_2  =poly.transform(X_test)
y_pred_2  = model.predict(X_test_2)

mae = mean_absolute_error(y_test,y_pred_2)
print("Mean Absolute Error: ",mae)
r2 = r2_score(y_test,y_pred_2)
print("R2 Score: ",r2)

# Code ends here


