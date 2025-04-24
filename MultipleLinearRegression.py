# An alternative and straightforward model for predicting movie revenue using linear regression with multiple inputs.
# Revenue = m1 * budget + m2 * runtime + m3 * release month

import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("movie_dataset.csv")
df['release_month'] = pd.to_datetime(df['release_date']).dt.month

reg = linear_model.LinearRegression()
reg.fit(df[['budget','runtime','release_month']], df.revenue)

userBudget = int(input("Provide an integer budget for your movie: "))
userRuntime = int(input("Provide an integer runtime in minutes for your movie: "))
userMonth = int(input("Provide an integer month for your movie: "))

user_input_df = pd.DataFrame([[userBudget, userRuntime, userMonth]],
                             columns=['budget', 'runtime', 'release_month'])
prediction = reg.predict(user_input_df)

print(f"Your movie is predicted to generate ${prediction[0]:,.2f}.")
