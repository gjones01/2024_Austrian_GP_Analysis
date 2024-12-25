Objective:
- Analyze the sector times, fastest laps and standard deviation of the drivers in the 2024 Austrian GP.
    Addtionally: Predict George Russell's sector 3 time on lap 41.
- Create visuals showcasing the extracted data and drawing an analysis form that

Dataset:
- The data set used in this project is in a csv format with lap data for each driver.

Skills Shown:
- Python
- Ability to utilize _sqlite3 to query data to extract specific data needed for analysis.
- Utilization of Panadas, MatplotLib and Seaborn for data manipulation and visualization.
- Basic knowledge of the sklearn libraries and OneHotEncoder for creating Machine Learning models.
 IDE: Visual Studios Code

 ## How to Run
1. Install dependencies: `pip install -r requirements.txt`.
2. Run `main.py` to generate graphs and insights.

Results:
Our first function titled: clean_data(line 9) takes our .csv and automates the cleaning process for us rather than manually cleaning each time. This structure allows for very simple workflow. I have access to the exact same csv format for other races. Therefore preprocessing them will be as simple as executing the function!


In the next function titled: query_calculation (starting at line 23), I was able to query data using _sqlite3 to see the top 5 drivers
with the fastest average lap times, as well as average sector times:

Average Sector Time & LapTime For Each Driver(in seconds):
  Driver     Avg_S1     Avg_S2     Avg_S3  Avg_LapTime
0    RUS  18.092586  31.849887  21.328056    71.307014
1    PIA  18.189686  31.743014  21.346986    71.333859
2    SAI  18.195543  31.669380  21.463718    71.370859
3    HAM  18.281700  31.803676  21.507620    71.632958
4    VER  18.587900  31.608028  21.474169    71.690859

As you can see George Russell had the fastest average lap times for the entire race. He would go on to win due to the 2 leaders (Norris & Verstappen) crashing into each other.

The second query (starting at line 43) extracts the fastest lap from the top 5 drivers and ranks them in ascending order:

Faster Lap Per Driver(in seconds):
  Driver  Fastest Lap
0    ALO       67.694
1    VER       67.719
2    NOR       68.016
3    PIA       68.697
4    RUS       69.164

As you can see, Fernando Alonso had the quickest lap of the race. This is a relevant statistic due to the fact that whoever gets the fastest lap of a race, will receive an additonal point regardless of finish position.


The final query (line 57) simply isolates the drivers and their laptimes. The reason being is I wanted to look at the standard deviation for the top 5 most consistent drivers (with respect to lap time). Querying that data out and converting it into a data frame allows for the use of Pandas to calculate the standard deviation. Therefore the result is:

Standard Deviation of Lap Times Per Driver(in seconds)
   Driver  Lap Consistency
7     MAG         3.363140
13    RUS         3.432562
5     HUL         3.440655
12    RIC         3.441535
11    PIA         3.446521

In this case you can see the top 5 most consistent lap time drivers are very similar. This is an interesting statistic to look at personally as it can provide some insight on why certain drivers may vary in times compared to others. To dive deeper into this why I will need to acquire some telemetry data. By calling this function we can display all of these query results.

Since the queried data was displayed, I created some visualizations. In this first case I wanted to see the standard deviation in laptimes for the drivers, as seen in this image ![Lap Standard Deviation](2024_AustrianGP_Project/LapSTD.png). On the x-axis we have the drivers and on the y-axis we have the standard deviation of all of the laps. In other words, this shows how consistent their lap times were with lower deviations being more consistent and higher deviations being less consistent. An interesting point in this graph (that I mentioned earlier) is Max Verstappen and Lando Norris having higher deviations compared to the other drivers. On lap 64  of the race, they collided with each other resulting in Verstappen having a punctured wheel, slowly crawling to the pits while Norris eventually had to retire a damamged car. This incident significantly affected their lap times thus resulting in this larger standard deviation.

In the next graph I have a heat map (using Seaborn) where you can see the fastest and slowest sectors of the track for the drivers.
![Heat Map of Sectors](2024_AustrianGP_Project/SectorHeatMap.png). What this map is telling us is simply which sectors of the track have the shortest times and which have the longest. On the x-axis is the average sector times for each driver and the y-axis identifies the driver at hand. Dark blue indicates the drivers had the fastest times in Sector 1. Inside the dark blue are variations in shade with slightly lighter shades of blue indicating slightly slower average sector times. This makes sense since at this particular circuit (Red Bell Ring) Sector 1 is coming out of Sector 3. This is a long straight on the track where the drivers will be carrying a lot of speed down into turn 1 which is a heavy braking zone, followed by another straight. Sector 2 is the slowest portion of the circuit due to more braking zones and corners. Therefore, it is indicated as red on the heatmap. Finally, sector 3 is the second fastest sector of the circuit with it being blue but not as dark. I have provided an official diagram of the track provided by Formula 1. Please note the colors indicating the sectors on this Formula 1 diagram having no relation to the color code on the heat map. It is strictly for discriminating the sectors. Red indicates sector 1, blue being sector 2 and yellow being sector 3. ![Red Bull Ring](2024_AustrianGP_Project/RedBullRing.png)

The final visual pulled from querying data is a simple stem graph ranking the top 5 drivers with their fastest lap. As mentioned before in the query, Fernando Alonso had the fastest lap of the entire race, hence he is ranked #1, followed by the other drivers. This is a very simple visualization, but great for a quick reference of who had the fastest lap. ![Fastest Lap](2024_AustrianGP_Project/FastestLapStem.png)

Linear Regression Model:
I wanted to play around with a basic Machine Learning model so I created a Linear Regression Model(Starting at line 149). With my feature variables as the lap number, compound, tyre life, Sector 1 and Sector 2 times, I set the target variable the Sector 3 times. In conjunction I wanted the model specifically predict his sector 3 time on lap 41, therefore the model learned from the 40 previous laps. This was the output:

Sector3Time Prediction for George Russell (41st Lap):
Actual Sector3Time: 21.451 seconds
Predicted Sector3Time: 21.503939084427355 seconds
Mean Absolute Error (MAE): 0.05293908442735429 
Mean Squared Error (MSE): 0.0028025466600065455

Along with this output is a simple bar graph comparing the actual sector 3 compared to the predicted sector 3 time. ![Model Prediction](2024_AustrianGP_Project/Sector3Prediction.png)

Considering this is just a very simple model it predicted George Russell's laptime to roughly 0.05 seconds, which in Formula 1 is by no means a negligible difference. However it is still fairly accurate. It is important to keep in mind that there are many factors when it comes to prediciting a sector time for a Formula 1 car, down to how much energy is being deployed from the car's battery to increase or decrease power output. However, this can still provide some good insights and I look forward to building more complicated models. My next goal is to acquire car telemetry data to analyze a particular driver's driving style and how they can optimize their lap times. Make sure to follow me on GitHub and to stay up to date with my projects!




