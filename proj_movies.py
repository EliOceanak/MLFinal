import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

movies = pd.read_csv('movies2.csv')

movies = movies.iloc[:,1:]

cols = list(movies.columns)

a,b = cols.index("avg_vote"), cols.index("year")

cols[b], cols[a] = cols[a], cols[b]

movies = movies[cols]

corr = movies.corr()

sns.set_style(style= "white")

f, ax = plt.subplots(figsize=(11,9))

cmap = sns.diverging_palette(10,250, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True, annot=True)

plt.title("Correlation Heatmap Movies")

plt.show()

sns.scatterplot(data=movies, x="metascore", y="avg_vote")

plt.title('Metascore vs Avg vote')

plt.show()

sns.scatterplot(data=movies, x="votes", y="avg_vote")

plt.title('Number of Votes vs Avg vote')

plt.show()


sns.scatterplot(data=movies, x="duration", y="avg_vote")

plt.title('Duration vs Avg vote')

plt.show()

#splitting dataset

y = movies['avg_vote']

x = movies[['votes', 'duration', 'metascore']]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

mlr = LinearRegression()

mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))

y_predicted = mlr.predict(x_test)

from sklearn import metrics

meanSquare = metrics.mean_squared_error(y_test, y_predicted)
meanAbsolute = metrics.mean_absolute_error(y_test, y_predicted)


print('mean squared error: ', meanSquare)
print('mean absolute error: ', meanAbsolute)


x = movies[['votes', 'duration']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

mlr = LinearRegression()

mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
print(list(zip(x, mlr.coef_)))

y_predicted = mlr.predict(x_train)

meanSquare = metrics.mean_squared_error(y_train, y_predicted)
meanAbsolute = metrics.mean_absolute_error(y_train, y_predicted)

y_predicted = mlr.predict(x_test)

print('Testing data mean squared error: ', meanSquare)
print('Testing data mean absolute error: ', meanAbsolute)


meanSquare = metrics.mean_squared_error(y_test, y_predicted)
meanAbsolute = metrics.mean_absolute_error(y_test, y_predicted)


print('Testing data mean squared error: ', meanSquare)
print('Testing data mean absolute error: ', meanAbsolute)


