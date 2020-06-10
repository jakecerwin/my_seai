from kafka import KafkaConsumer
import csv

users = set()

# returns string of title given a ConsumerRecord
def parse_cr(cr):
    binary = cr.value
    string = binary.decode('utf-8')
    # [time, user id, GET request]
    return string.split(',')


# returns string of title given a ConsumerRecord in name+name+year format regardless of rate or data
def get_title(cr):
    get = parse_cr(cr)[2]
    head = get[5:9]
    if head == 'data':
        trunc = get[12:]
        return trunc.split('/')[0]
    else:
        trunc = get[10:]
        return trunc.split('=')[0]


def gather_popularity():
    date = None
    first = None
    popularity = dict()
    days = set()

    consumer = KafkaConsumer(
        'movielog',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        group_id='jcerwin-stream',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )

    for message in consumer:
        if first is None:
            first = message
        else:
            if message == first:
                print("repeat")
                break

        parsed = parse_cr(message)
        r_block = parsed[2]
        head = r_block[5:9]
        # look for watches only not reviews
        if head == 'data':
            trunc = r_block[12:]
            title = trunc.split('/')[0]

            minutes = r_block.split('/')[4][:-4]
        else:
            continue

        new_date = (parsed[0])[5:10]
        print(new_date)

        # If stream has finished calculate views per month
        if date is not None:
            if int(new_date[0:2]) < int(date[0:2]) or \
               (int(new_date[0:2]) == int(date[0:2]) and int(new_date[3:]) < int(date[3:])):
                #print(len(days))
                wp_month = dict()
                for title in popularity:
                    wp_month[title] = popularity[title] / len(days)
                return wp_month

        date = new_date
        if date not in days:
            days.add(date)

        user = parsed[1]
        if user not in users:
            users.add(user)

        if title in popularity:
            if int(minutes) == 0:
                viewed = popularity[title]
                popularity.update({title: viewed + 1})
        else:
            popularity.update({title: 1})

    return RuntimeError

def gather_titles():
    consumer = KafkaConsumer(
        'movielog',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        group_id='jcerwin',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )

    f = open("movie_titles.txt", "r")
    fl = f.readlines()
    f.close()
    s = set(fl)
    i = len(s)

    f = open("movie_titles.txt", "a")
    for message in consumer:
        if i > 27000:
            break
        title = get_title(message) + '\n'
        if title in s:
            continue
        else:
            s.add(title)
            f.write(title)
            i = i + 1

    f.close()

#with open('views.csv', 'w') as csv_file:
#    writer = csv.writer(csv_file)
#    for key, value in gather_popularity().items():
#        writer.writerow([key, value])


results = dict()
for i in range(10):
    print(i)
    temp = gather_popularity()
    print("here")
    for item in temp:
        if item in results:
            results[item] += temp[item]
        else:
            results[item] = temp[item]


for item in results:
    results[item] = results[item] * 1000000 / len(users)

print(len(users))

print(results['jurassic+park+1993'])

with open('views.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results.items():
        writer.writerow([key, value])


