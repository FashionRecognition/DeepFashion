from pymongo import MongoClient

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

word_frequencies = {}

# Select all titles
for record in db.ebay.find({}, {"title": 1, "_id": 0}):

    # Break each title into individual words
    for word in record['title'].split(" "):

        # Build frequency dictionary
        if word.lower() in word_frequencies:
            word_frequencies[word.lower()] += 1
        else:
            word_frequencies[word.lower()] = 1

#
for key, value in sorted(word_frequencies.items(), key=lambda kv: (kv[1], kv[0])):

    # Cull out some of the trash
    if len(key) > 1 and key.isalpha():
        print(key + ": " + str(value))

print(len(word_frequencies))
