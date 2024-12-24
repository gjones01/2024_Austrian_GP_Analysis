import pandas as pd
import _sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#I created a function to automate the cleaning and conversion of my data into a data frame and connecting the SQL database
#This searches for missiing values and allows me to modifiy if needed
def clean_data(csv_file='/Users/gerryjr/Desktop/Austrian_Grand_Prix.csv'):
    df = pd.read_csv('/Users/gerryjr/Desktop/Austrian_Grand_Prix.csv')
    
    conn = _sqlite3.connect('AustrianGP.db')
    df.to_sql('Race_Data', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    print(df.isnull().sum())
    df['Sector1Time'] = df['Sector1Time'].fillna('No Data')
    print(df.head())

    return df

#Now that I have the data cleaned I want to create a data base and table to start applying query operations to extract specific data
def query_calculations(csv_file='/Users/gerryjr/Desktop/Austrian_Grand_Prix.csv', db_name='AustrianGP.db', table_name='Race_data'):
    conn = _sqlite3.connect('AustrianGP.db')
    
    #This first query is calculating the average sector time (1st, 2nd & 3rd) for each driver. This can be useful for
    #overall statistics in which driver had the fastest average sectors at this grand prix
    query_avg_sector= '''
    SELECT Driver, AVG(Sector1Time) AS Avg_S1, AVG(Sector2Time) AS Avg_S2, AVG(Sector3Time) AS Avg_S3, AVG(LapTime) AS Avg_LapTime
    FROM Race_Data
    GROUP BY Driver
    ORDER BY Avg_LapTime ASC;
    '''
    #We now execute the query, converting it into a data frame and print the results! Most times we are interested 
    #in the top few drivers so we isolate the print with a .head() to see the top 5 drivers with the fastest average sector times.
    df_avg_sector = pd.read_sql(query_avg_sector, conn)
    print('Average Sector Time & LapTime For Each Driver')
    print(df_avg_sector.head())

    #The next query in this function is isolating the fastest laps by each driver and ranking them accordingly.
    #This is another usefull statistic as the driver who acquires the fastest lap is awarded an extra point regardless
    #of the position they finish the race!
    query_fastest_lap = '''
    SELECT Driver, MIN(LapTime) AS "Fastest Lap"
    FROM Race_Data
    GROUP BY Driver
    ORDER BY "Fastest Lap" ASC
    '''
    #Again we execute and convert the query to a data frame and print the top 5. In the output we can see that Fernando Alonso
    #clocked the fastest lap of the race.
    df_fastest_lap = pd.read_sql(query_fastest_lap, conn)
    print('Faster Lap Per Driver:')
    print(df_fastest_lap.head())

    #This next query is interesteing as I wanted to look at the variance (or standard deviation), so first I needed to simply extract
    #extract the drivers and their corresponding lap times.
    query_STD = '''
    SELECT Driver, LapTime
    FROM Race_Data;
    '''
    
    #Again we execute and convert the query to a data frame while also applying .std() to see such standard deviation. This is
    #why it's important to convert the queries to data frames because it allows for furhter operations that may utilize other libraries.
    #In this case Pandas is used to calculate the standard deviation
    laptimes = pd.read_sql(query_STD, conn)
    stddev_laptimes = laptimes.groupby('Driver')['LapTime'].std().reset_index()
    stddev_laptimes.columns = ['Driver','Lap Consistency']
    stddev_laptimes.sort_values(by='Lap Consistency', inplace=True)
    print('Standard Deviation of Lap Times Per Driver')
    print(stddev_laptimes.head())
    conn.close()

    return {
        'average sector':df_avg_sector,
        'fastest lap':df_fastest_lap,
        'driver consistency':stddev_laptimes
    }
#We created a dictionary of the corresponding results, that way when we want to call the function later. we can simply insert the key
#for the desired results we want to look at. I could have written a seperate function for each query operation but I prefer
#this more uniform setup where I can simply use a dictionary to pick what I want to analyze.
results = query_calculations()
df_avg_sector = results['average sector']
df_fastest_lap = results['fastest lap']
stddev_laptimes = results['driver consistency']

#This is calling the query function to display all of the results from the 3 calculations we did.
query_calculations('AustrianGP.db')

#Now that we have the extracted data we wantto look at, it's time to visualize it in a variety of forms.
#This first function is a scatter plot of the standard deviation of the laps for each driver. It is in ascending order
#to see the most consistent to the least consistent lap times.
def plot_deviation(stddev_laptimes):
    conn = _sqlite3.connect('AustrianGP.db')
    plt.figure(figsize=(10, 6))
    plt.scatter(stddev_laptimes['Driver'],stddev_laptimes['Lap Consistency'], color='red', edgecolor='black')
    plt.xlabel('Driver')
    plt.ylabel('Driver Lap Consistency')
    plt.title('Driver Lap Consistency')
    plt.show()
    conn.close()

#To visualize the plot we simply call the function.
plot_deviation(stddev_laptimes)

#This next function utilizes Seaborn to create a heatmap of the fastest and slowest sectors for each driver.
#This will provide an idea of which sectors are quickest amongst the drivers and who tends to be slightly faster.
def plot_avg_sectors(df_avg_sector):
    conn = _sqlite3.connect('AustrianGP.db')
    sector_times = df_avg_sector[['Avg_S1','Avg_S2','Avg_S3']]
    sector_times.index = df_avg_sector['Driver']
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(sector_times, annot=True, cmap='coolwarm', cbar_kws={'label':'Times'})
    ax.set_xlabel('Times',fontsize=10)
    ax.set_ylabel('Drivers',fontsize=8)
    plt.title('Heatmap Of Average Sector Times')
    plt.show()
    conn.close()

#To visualize it we call thefunction again.
plot_avg_sectors(df_avg_sector)

#Now we want to see a simple stem graph of who had the fastest lap. In this case we want to take the top 5 drivers and plot their
#fastest laptime. This is something to show an individual who may not be familiar with the sport or python in general
#and you want to show them a simple graph of who had the fastest lap.
def plot_fastest(df_fastest_lap):
    conn = _sqlite3.connect('AustrianGP.db')
    df_fastest_lap = df_fastest_lap.sort_values(by='Fastest Lap').head(5)

    plt.figure(figsize=(10,8))
    plt.stem(df_fastest_lap['Driver'], df_fastest_lap['Fastest Lap'], basefmt=" ", linefmt=('green'), markerfmt='o', label='Fastest Lap')
    plt.ylim(60,70)


    plt.title('Fastest Lap For Each Driver')
    plt.xlabel('Drivers')
    plt.ylabel('Lap Time')
    plt.show()
    conn.close()
#Again we call the function to visualize it.
plot_fastest(df_fastest_lap)

#Now shifting gears a bit(no pun intended), I was curious to create a Linear Regression Machine Learning model to specifically
#predict George Russell's Sector 3 time on lap 41 ( a completely random lap).

# NOTE: it is important to understand there are much more variables that account for lap time, such as barometric pressure,
#temperature of the track, temperature of the tyres, wind speed/direction. There are also variables involving the physical
#telemetry of the car, some of which is not publically available yet or at all. However, I thought it would be interesting to
#see how accurate (or not) a simple ML model would be at predicting a specific sector time on a specific lap.

#This imports all of the necessary libraries for the model being used.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# As usual we write in the csv as a data frame
df = pd.read_csv('/Users/gerryjr/Desktop/Austrian_Grand_Prix.csv')

# We want to specifically look at George Russell for this model so we clarify that in this variable.
driver_data = df[df['Driver'] == 'RUS'].sort_values(by='LapNumber')

# We want to remove the laps that don't have sector times (the first lap has no sector 1 time)
driver_data = driver_data.dropna(subset=['Sector1Time', 'Sector2Time', 'Sector3Time', 'TyreLife'])

# Now split the data into training (first 40 laps) and testing (41st lap). Remember the goal here is to predict his 41st Sector 3 time. So we will trian
#the model on the first 40 laps
train_data = driver_data[driver_data['LapNumber'] <= 40]
test_data = driver_data[driver_data['LapNumber'] == 41]

# Features and targets are specified here. I have included the the tyre compound as a feature varialbe for any future predictions that involve stints executed on
#multiple tyre compounds
X_train = train_data[['LapNumber', 'Compound', 'TyreLife', 'Sector1Time', 'Sector2Time']]
y_train = train_data['Sector3Time']

X_test = test_data[['LapNumber', 'Compound', 'TyreLife', 'Sector1Time', 'Sector2Time']]
y_test = test_data['Sector3Time']

# OneHotEncoder is exclusive for the tyre compound column since it is a string. This will convert it into a binary format for if you wanted to do something
#along the lines of predicting Rusell's 60th lap for example.
encoder = OneHotEncoder(sparse_output=False)
compound_encoded_train = encoder.fit_transform(X_train[['Compound']])
compound_encoded_test = encoder.transform(X_test[['Compound']])

#Now it is converted to a data frame and re-added to the features.
compound_encoded_train_df = pd.DataFrame(compound_encoded_train, columns=encoder.get_feature_names_out(['Compound']))
compound_encoded_test_df = pd.DataFrame(compound_encoded_test, columns=encoder.get_feature_names_out(['Compound']))
X_train = pd.concat([X_train.drop('Compound', axis=1).reset_index(drop=True), compound_encoded_train_df.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test.drop('Compound', axis=1).reset_index(drop=True), compound_encoded_test_df.reset_index(drop=True)], axis=1)

# Now it is time to train the model with linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# This is the prediction variable
y_predict = model.predict(X_test)

# This is important to actually evaluate how accurate the model is.
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("Sector3Time Prediction for George Russell (41st Lap)")
print(f"Actual Sector3Time: {y_test.values[0]}")
print(f"Predicted Sector3Time: {y_predict[0]}")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Using matplotlib, this visualizes the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.bar(['Actual Sector3Time', 'Predicted Sector3Time'], [y_test.values[0], y_predict[0]], color=['blue', 'orange'])
plt.title('Sector3Time Prediction vs Actual (41st Lap)')
plt.ylabel('Sector3Time (seconds)')
plt.grid(axis='y')
plt.show()

#NOTE: I plan on continuing this project with more complex machine learning models and eventually incorporating other 
#varialbes that I mentioned before to draw even more precise predictions. I figured it would be fun to try this regression model
#and showcase some basic skills I have in Machine Learning. Make sure to follow me on GitHub to get updates on this project!