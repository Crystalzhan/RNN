# Multivariate Temperature Forecasting using RNN/LSTM

## Objective 
The major task of this project is to build and train an optimal LSTM model for multivariate time series forecasting. The dataset is provided by the Weather Station at the Max Planck Institute for Biogeochemistry in Jena, it contains hourly weather conditions from 2009 to 2016. Here we use data from 01.01.2009 01:00:00 to 31.12.2014 23:00:00 to train the model, and then use the model to make prediction on the rest of the data. The objective of this project is to predict the temperature of the next hour given the climate conditions and temperature over the last 24 hours.

## Table of Contents
1. Project and Data overview
2. Imports packages and data
3. Exploratory Data Analysis (EDA)
   - Description Analysis
   - Visualization Analysis
4. Data Preprocessing
   - Normalize Data
   - Concatenate training data to Keras RNN/-ready
   - Split training data and validation data
   - Reshape Input data
5. Model Building
   - Build Base Model
   - Train Models
   - Optimal Models performance visualization
   - Define final model
6. Predictions

## 1. Project and Data overview

The problem is a multivariate forecasting problem. Given a historical data of climate conditions, you predict the temperature of the next hour based on the climate conditions and temperature over the last 24 hours.

### Data

The dataset contains weather conditions recorded at the Weather Station at the Max Planck Institute for Biogeochemistry in Jena, Germany, over several years. In this dataset, 14 different quantities (including air temperature, atmospheric pressure and humidity) were recorded every 10 minutes.

The original data goes back to 2003, but we use a subset for this competition, from 2009 to 2016 (both inclusive). Also for our competition, we use a further reduced version in which recordings are kept for every hour.

## 2. Imports packages and data
Package include: numpy, pandas, matplotlib, seaborn, sklearn, keras

## 3. Exploratory Data Analysis (EDA)
### 3-A. Description Analysis
From description sheet, we can see the following facts:

- The 14 climate features are all float number with no missing value, the total number of instances in training set is 52556
- The scale of each feature are different, normalization of dataset is required at data preprocessing step
- Skewness can be observed by compare median (50%) and mean, and variable's distribution can be visualized in following histogram.

The Histogram and Density plots show the following distribution pattern:

- Unimodal, close to normal distribution: T and Tpot
- Unimodal, skew to the left: p, Tdew, rh
- Unimodal, skew to the right: VPmax, VPact, VPdef, sh, H2OC, rho, wv, max.wv
- Bimodal: wd

From time series plot, we can see the following 10 attributes show significant seasonality: T(degC), Tpot(K), Tdew(degC), rh(%), VPmax(mbar), VPact(mbar), VPdef(mbar), sh(g/kg), H2OC(mmol/mol), rho(g/m**3)

Next, we further explore the seasonality of Temperature with box plot, group the data by month and hour to display the distribution of each group.
- First we group the data by month, to visulize yearly seasonality, the boxplot confirm the yearly seasonality that we saw in earlier line plot and provide some additional insight:
  - Temperature is highest in July and lowest in January
  - The distribution of temperature have more outliers during summer
- Second, we group the temperature by hour of the day, to explore daily seasonality. From plot above we can see as what we expected, the temperature is slightly higher at noon than at night and morning. There are also less outlier in the middle of the day.

## 4. Data Preprocessing
### 4-A. Normalize Data
Since the dot product (i.e., weighted sum) is susceptible to values of different magnitudes and/or ranges. We need normalize data before training model. Here we will use Standard Normal (Z-value) conversion: subtract the mean of each variable and divide by the standard deviation.

### 4-B. Concatenate training data to Keras RNN/-ready
Our forecasting scheme is to look back 24 hours (e.g. h1 through h24) and predict the next hour (h25). And we use all 14 climate measurements of the previous 24 hours to predict the temperature (only) of the 24th hour (thus just one feature). The dataset -- xtrain and ytrain are as follow:

- xtrain:
  - Shape: 52542 x 336 (where 52542 = 52566 - 24, 336 = 14 * 24)
  - The first entry in xtrain spans from 01.01.2009 01:00:00 to 02.01.2009 00:00:00
  - The last entry in xtrain spans from 30.12.2014 23:00:00 to 31.12.2014 22:00:00
- ytrain:
  - Shape: 52542 * 1
  - The first entry in ytrain is the temperature of 02.01.2009 01:00:00
  - The last entry in ytrain is the temperature of 31.12.2014 23:00:00


### 4-C. Split training data and validation data
Split the training data into 2 parts: the first 5 year for training predictive models and the final year for evaluating models.

- x_valid:
  - Shape: 8760 x 336 (where 8760 = 365 * 24)
  - Entry from 01.01.2014 01:00:00 to 31.12.2014 23:00:00
- y_valid:
  - Shape: 8760 x 1
  - Entry from 01.01.2014 01:00:00 to 31.12.2014 23:00:00
- x_train:
  - Shape: 43782 x 336 (where 43782 = 52542 - 8760)
  - Entry from 01.01.2009 01:00:00 to 31.12.2013 23:00:00
- y_train:
  - Shape: 43782 x 1
  - Entry from 01.01.2009 01:00:00 to 31.12.2013 23:00:00

### 4-D. Reshape Input data
Reshape training, validation and testing input data into 3D format expected by LSTMs, namely [samples, timesteps, features]

## 5. Model Building
### 5-A. Build Base Model
In this step, I define a Vanilla LSTM model as the base model.

- Single hidden LSTM layer with 50 units, the input shape will be 24 time steps with 14 features
- Then, an output layer with 1 element
- We will use MAE (Mean Absolute Error) loss function and the efficient Adam version of stochastic gradient descent
- The model will be fit for 50 training epochs with a batch size of 1000

### 5-B Train Models
#### Train model by:
- Adding stacked LSTM
- Changing LSTM layer to bidirectional LSTM
- Adding CNN layer
- Increading nodes in LSTM layer
- Decreasing batch size
- Increasing epochs number
- Adding recurrent drop out rate

#### Observations:

- t9_model training and testing loss are much higher than the rest 2 models, and it performance fluctuate over epochs, indicate its learing process is not stable
- t7_model and t8_model performance are close to each other, both of them have significant overfitting issue after epoch 100, will add drop out rate to eliminate this impact in next step

### 5-C. Optimal Models performance visualization
### 5-D. Define final model
Based on the plot from 5-C, we can also see that t11_model did a better job, it has lower and less fluctuate loss rate. So here I choose t11_model as my final model to make prediction on testing data.

The final model has single bidirectional LSTM layer with 200 units, and followed by an output layer with 1 element, it fit for 100 epochs with a batch size of 250. From its loss plot in figure 4, we can tell it has a relative stable decreasing loss curve, and the difference between training and validation is small; all these indicating t11_model is a stable model with lower overfitting risk, the score returned from Kaggle also support my idea. 

## 6. Predictions
Use the final model to predict from "02.01.2015 00:00:00" (Jan 2, 2015 0:00 am) to "01.01.2017 00:00:00" (Jan 1, 2017 0:00 am)

**Final MAE dropped from base model's 1.8820 to 0.5470.**

## Limitations and Future Work
Some limitations exist in this project, even though stationary is observed in EDA section, I did not take seasonality and stationary into consideration when building model. In future work, I will try to build models after remove these factors. 

