Objective:
- Analyze the sector times, lap times and standard deviation of the drivers in the 2024 Austrian GP.
    Addtionally: analyze George Russell's lap time degradation using Linear Regression Machine Learning (not anticipated to be very accurate)
- Create visuals showcasing the extracted data and drawing an analysis form that

Dataset:
- The data set used in this project is in a csv format with lap data for each driver.

Skills Shown:
- Python
- Ability to utilize _sqlite3 to query data to extract specific data needed for analysis.
- Utilization of Panadas, MatplotLib and Seaborn for data manipulation and visualization.
- Basic knowledge of the sklearn library and OneHotEncoder for creating Machine Learning models.
 IDE: Visual Studios Code

Results:
Our first function titled: clean_data(line 9) takes our .csv and automates the cleaning process for us rather than manually cleaning each time. This structure allows for very simple workflow if I wanted to analyze data from another race.


In the function titled: query_calculation (starting at line 23), I was able to query data to see the top 5 drivers
with the fastest average lap times, as well as average sector times:

Average Sector Time & LapTime For Each Driver
  Driver     Avg_S1     Avg_S2     Avg_S3  Avg_LapTime
0    RUS  18.092586  31.849887  21.328056    71.307014
1    PIA  18.189686  31.743014  21.346986    71.333859
2    SAI  18.195543  31.669380  21.463718    71.370859
3    HAM  18.281700  31.803676  21.507620    71.632958
4    VER  18.587900  31.608028  21.474169    71.690859

As you can see George Russell had the fastest average lap times for the entire race. He would go on to win due to the 2 leaders (Norris & Verstappen) crashing into each other.

The second query (starting at line 43) extracts teh fastest lap from the top 5 drivers and ranks them in ascending order:

Faster Lap Per Driver:
  Driver  Fastest Lap
0    ALO       67.694
1    VER       67.719
2    NOR       68.016
3    PIA       68.697
4    RUS       69.164

As you see Fernando Alonso had the quickest lap of the race. This is a relevant statistic due to the fact that whoever gets the fastest lap of a race, will receive an additonal point regardless of finish position.


The final query (line 57) simply isolates the drivers and their laptimes. The reason being is I wanted to look at the standard deviation for the top 5 most consistent drivers (with respect to lap time). Querying that data out and converting it into a data frame allows for the use of Pandas to calculate the standard deviation. Therefore we received this result:

Standard Deviation of Lap Times Per Driver
   Driver  Lap Consistency
7     MAG         3.363140
13    RUS         3.432562
5     HUL         3.440655
12    RIC         3.441535
11    PIA         3.446521

By calling this function we can display all of these query results.

Since the queried was displayed, I created some visualizations. In this first case I wanted to seeing the standard deviation in laptimes for the drivers, as seen in this image ![Alt Text](/Users/gerryjr/Desktop/LapSTD.png)

Predicting Lap Time Degradation (George Russell):


