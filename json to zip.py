import gzip
import json

# Load the JSON data
with open("live_conversation.json", "r") as f:
    data = json.load(f)

# Compress the data and write it to a gzip file
with gzip.open("data.json.gz", "wb") as f:
    f.write(json.dumps(data).encode())



# import json

# # load the JSON data from file
# with open("live_conversation.json", "r") as json_file:
#     data = json.load(json_file)

# # write the data to a text file
# with open("output.txt", "w") as text_file:
#     text_file.write(json.dumps(data, indent=10))


# import json
# import os

# import json

# # Load the JSON data from the file
# with open("live_conversation.json", "r") as json_file:
#     json_data = json_file.read()

# # Parse the JSON data
# try:
#     data = json.loads(json_data)
# except json.decoder.JSONDecodeError as e:
#     print(f"JSONDecodeError: {e}")


# # # Write the text data to the file
# # with open("output.txt", "w") as text_file:
# #     text_file.write(json.dumps(data, indent=4))
import csv

# Read the contents of the CSV file
with open("./data/first_2M_final.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    
    # Write the contents of the CSV file to a text file
    with open("output.txt", "w") as text_file:
        for row in reader:
            text_file.write("\t".join(row) + "\n")

