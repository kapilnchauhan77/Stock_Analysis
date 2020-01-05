# Stock_Analysis
This script let's you download and analyse the information for the stock you want with a minor forecast using Random Forest Generator.


To use it:

1) Download python 3.7 from https://www.python.org/downloads/
2) Make sure to add python to environment variables.
3) Type "pip install -r requirements.txt" in Command Prompt.
4) Run the script.
5) Pass the ticker of the stock you want to analyse.


The following is the result:

1) The data of that stock will get downloaded and information on the stock split will be provided.
![Split](https://user-images.githubusercontent.com/44964331/71069203-3d95d380-219e-11ea-91a6-0cb6e9f6c33d.png)

2) After this an interactive graph of the stock with respect to time will be provided.
![All_plots](https://user-images.githubusercontent.com/44964331/71069214-438bb480-219e-11ea-9b2c-08461883aa82.png)
This graphs includes the plot of Open, high, low, close and moving average of the stock over the years.

3) After closing this, a Candle stick graph with volume of stocks sold will be provided.
![Candle_Stick](https://user-images.githubusercontent.com/44964331/71069403-a9783c00-219e-11ea-8ef3-8b14b12e126b.png)
This is also interactive.
![Candle_stick_zoomed](https://user-images.githubusercontent.com/44964331/71069289-6b7b1800-219e-11ea-8462-730ca4e61527.png)

4) At last, a 7 day prediction will be plotted.
![Forecast](https://user-images.githubusercontent.com/44964331/71069460-cc0a5500-219e-11ea-94f3-9ae0e6b88a4f.png)
