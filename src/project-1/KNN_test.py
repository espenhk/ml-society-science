import pandas
features = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
        "Magnesium", "Total phenols", "Flavanoids", 
        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
        "Hue", "OD280/OD315 of diluted wines", "Proline"]
target = 'Class'

df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names=[target] + features)
print(type(df))
print(df.head())

