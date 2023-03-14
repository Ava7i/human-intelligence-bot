import csv

# Read the contents of the CSV file
with open("./data/first_2M_final.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    
    # Write the contents of the CSV file to a text file
    with open("output.txt", "w") as text_file:
        for row in reader:
            text_file.write("\t".join(row) + "\n")
