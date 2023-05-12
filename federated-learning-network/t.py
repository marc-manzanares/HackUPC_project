import json

# Data to be written
dictionary = {
	"name": "sathiyajith",
	"rollno": 56,
	"cgpa": 8.6,
	"phonenumber": "9976770500"
}

a = 'http://147.83.58.184:5001/'

split_url = str(a[:-1]).split(':')

# Serializing json
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
with open("model_"+split_url[-1]+".json", "w") as outfile:
	outfile.write(json_object)
