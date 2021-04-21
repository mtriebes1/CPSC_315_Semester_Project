from mysklearn.mypytable import MyPyTable

# Object Declaration
table = MyPyTable()

# Trims the Dataset (Gets Data Based on City)
city = "Sydney"
table.load_from_file("weatherAUS.csv")
table.column_names[0] = 'Location'
names, tables = table.group_by("Location")

city_index = names.index(city)

print("\n")
for i in range(10):
    print(tables[city_index][i])

table.data = tables[city_index]

table.save_to_file(city+"_weather.csv")