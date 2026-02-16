import csv
import random

with open('combined_microdoppler.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['label', 'value'])

    # Generate 150 rows for bird (label 0)
    for _ in range(150):
        value = round(random.uniform(0.3, 0.6), 3)
        writer.writerow([0, value])

    # Generate 150 rows for drone (label 1)
    for _ in range(150):
        value = round(random.uniform(0.6, 0.9), 3)
        writer.writerow([1, value])
print("created csv file")