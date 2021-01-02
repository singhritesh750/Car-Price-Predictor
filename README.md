import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

# Import data-set from external source:
path_ = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(path_, header=None)
headers = ["symbol_", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers

# Dealing with missing data in data_frame:
# 1.Counting missing values in each column
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# 2.now replacing the missing data "?" by NaN
df.replace("?", np.nan, inplace=True)  # replace "?" to NaN


# 2.(a)replace NaN by average of colunm/ Mean (for continuous variables)


def _func(col, d_type):  # for continuous variables
    a_v_g = df[col].astype(d_type).mean(axis=0)  # 2 lines for calculating avg of a column & then replacing NaN with it
    df[col].replace(np.nan, a_v_g, inplace=True)
    return a_v_g


avg_norm_loss = _func("normalized-losses", "float")
avg_bore = _func('bore', 'float')
avg_stroke = _func("stroke", "float")
avg_horsepower = _func('horsepower', 'float')
avg_peakrpm = _func('peak-rpm', 'float')

# 2.(b)replace NaN by Mode/frequency (for categorical variables)
# df['num-of-doors'].value_counts()
# ".idxmax()" method to calculate for us the most common type automatically
df["num-of-doors"].replace(np.nan, df['num-of-doors'].value_counts().idxmax(), inplace=True)

# 3.dropping will be done after using all the values to calculate average and other
df.dropna(subset=["price"], axis=0, inplace=True)  # simply drop whole row with NaN in "price" column
df.reset_index(drop=True, inplace=True)  # reset index, because we dropped two rows

# Data Formatting:Bringing data into common standard of expression for meaningful compare,Converting into required units
df['city-L/100km'] = 235 / df["city-mpg"]  # Convert mpg to L/100km by mathematical operation (235/mpg)
df["highway-mpg"] = 235 / df["highway-mpg"]  # convert mpg to L/100km by mathematical operation (235/mpg)
df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)  # rename column from "high-mpg" to "high-L/100km"

# Data Normalization: Normalize those variables so their value ranges from 0 to 1
# Using Simple feature Scaling method- replace (original value) by (original value)/(maximum value)
# df['length'] = df['length'] / df['length'].max()
# df['width'] = df['width'] / df['width'].max()
# df['height'] = df['height'] / df['height'].max()
# df[["length", "width", "height"]].head()  # show the scaled columns

# Data Binning: Binning to transforming continuous variables into discrete categorical 'bins', for grouped analysis.
df["horsepower"] = df["horsepower"].astype(int, copy=True)
plt.pyplot.hist(df["horsepower"])
# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# Creating bins to change continuous variable into categorical variable
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
df["horsepower-binned"].value_counts()

# dummy variables are used to convert categorical variable into continuous variable
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
df = pd.concat([df, dummy_variable_1], axis=1)  # merge data frame "df" and "dummy_variable_1"

# drop original column "fuel-type" from "df"
# df.drop("fuel-type", axis=1, inplace=True) # after using dummy variable best practice is to drop the original variable

# get indicator/dummy variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])
# change column names for clarity
dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

df = pd.concat([df, dummy_variable_2], axis=1)  # merge the new dataframe to the original datafram
# df.drop('aspiration', axis=1, inplace=True)  # drop original column "aspiration" from "df"
df.to_csv('clean_df.csv')
