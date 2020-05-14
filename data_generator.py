import csv
from numpy import random
from numpy import linspace
from matplotlib import pyplot
import pandas as pd

if __name__ == "__main__":
    fieldnames = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished', 
        'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 'ExternalReference', 'LocationLatitude', 
        'LocationLongitude', 'DistributionChannel', 'UserLanguage', 'Q1', 'Q3', 'Q4', 'message', 'message1', 'message2', 'messageToShow']
    csvfile = open('sample_data.csv', 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'StartDate': 'Start Date', 'EndDate': 'End Date', 'Status': 'Response Type', 'IPAddress': 'IP Address', 
        'Progress': 'Progress', 'Duration (in seconds)': 'Duration (in seconds)', 'Finished': 'Finished', 'RecordedDate': 
        'Recorded Date', 'ResponseId': 'Response ID', 'RecipientLastName': 'Recipient Last Name', 'RecipientFirstName': 'Recipient First Name', 
        'RecipientEmail': 'Recipient Email', 'ExternalReference': 'External Data Reference', 'LocationLatitude': 'Location Latitude', 
        'LocationLongitude': 'Location Longitude', 'DistributionChannel': 'Distribution Channel', 'UserLanguage': 'User Language', 
        'Q1': 'What is your favourite primary colour?', 'Q3': 'What is your favourite integer?', 'Q4': 'Enter a random string?', 
        'message': 'message', 'message1': 'message1', 'message2': 'message2', 'messageToShow': 'messageToShow'})
    writer.writerow({'StartDate': '{"ImportId":"startDate","timeZone":"America/Denver"}', 
        'EndDate': '{"ImportId":"endDate","timeZone":"America/Denver"}', 'Status': '{"ImportId":"status"}', 'IPAddress': '{"ImportId":"ipAddress"}', 
        'Progress': '{"ImportId":"progress"}', 'Duration (in seconds)': '{"ImportId":"duration"}', 'Finished': '{"ImportId":"finished"}', 
        'RecordedDate': '{"ImportId":"recordedDate","timeZone":"America/Denver"}', 'ResponseId': '{"ImportId":"_recordId"}', 
        'RecipientLastName': '{"ImportId":"recipientLastName"}', 'RecipientFirstName': '{"ImportId":"recipientFirstName"}', 
        'RecipientEmail': '{"ImportId":"recipientEmail"}', 'ExternalReference': '{"ImportId":"externalDataReference"}', 
        'LocationLatitude': '{"ImportId":"locationLatitude"}', 'LocationLongitude': '{"ImportId":"locationLongitude"}', 
        'DistributionChannel': '{"ImportId":"distributionChannel"}', 'UserLanguage': '{"ImportId":"userLanguage"}', 'Q1': '{"ImportId":"QID1"}', 
        'Q3': '{"ImportId":"QID3_TEXT"}', 'Q4': '{"ImportId":"QID4_TEXT"}', 'message': '{"ImportId":"message"}', 
        'message1': '{"ImportId":"message1"}', 'message2': '{"ImportId":"message2"}', 'messageToShow': '{"ImportId":"messageToShow"}'})
    row_template = {'StartDate': '2020-05-05 12:00:00', 'EndDate': '2020-05-05 12:00:00', 'Status': '0', 'IPAddress': '127.0.0.1', 
        'Progress': '100', 'Duration (in seconds)': '0', 'Finished': 'True', 'RecordedDate': '2020-05-05 12:00:00', 'ResponseId': '', 
        'RecipientLastName': '', 'RecipientFirstName': '', 'RecipientEmail': '', 'ExternalReference': '', 'LocationLatitude': '0.0', 
        'LocationLongitude': '0.0', 'DistributionChannel': 'anonymous', 'UserLanguage': 'EN', 'Q1': '', 'Q3': '', 'Q4': '', 
        'message': '', 'message1': 'We like larger integers and longer strings!', 'message2': "We don't like smaller integers or shorter strings.", 
        'messageToShow': ''}
    rows = []
    sigma = [chr(i) for i in range(32, 127)]
    #Red
    for i in range(500):
        row = dict(row_template)
        row['Q1'] = 'Red'
        row['message'] = i % 2 + 1
        if i % 2 == 0: #message1
            row['Q3'] = round(random.normal(45, 20))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(70, 20)))))
        else: #message2
            row['Q3'] = round(random.normal(90, 12))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(20, 13)))))
        rows.append(row)
    #Green
    for i in range(500):
        row = dict(row_template)
        row['Q1'] = 'Green'
        row['message'] = i % 2 + 1
        if i % 2 == 0: #message1
            row['Q3'] = round(random.normal(60, 15))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(70, 10)))))
        else: #message2
            row['Q3'] = round(random.normal(45, 15))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(50, 18)))))
        rows.append(row)
    #Blue
    for i in range(500):
        row = dict(row_template)
        row['Q1'] = 'Blue'
        row['message'] = i % 2 + 1
        if i % 2 == 0: #message1
            row['Q3'] = round(random.normal(45, 20))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(70, 16)))))
        else: #message2
            row['Q3'] = round(random.normal(15, 13))
            row['Q4'] = "".join(random.choice(sigma, max(1, round(random.normal(90, 10)))))
        rows.append(row)
    random.shuffle(rows)
    for row in rows:
        writer.writerow(row)
    csvfile.close()

    sample = pd.read_csv('sample_data.csv')
    sample = sample[2:]
    sample_columns = ['Q1', 'Q3', 'Q4', 'message']
    sample = sample[sample_columns]
    sample['Q3'] = pd.to_numeric(sample['Q3'])
    sample['length'] = sample['Q4'].str.len()
    sample['message_2'] = pd.to_numeric(sample['message']) - 1
    sample['colour_green'] = (sample['Q1'] == 'Green').astype(int)
    sample['colour_blue'] = (sample['Q1'] == 'Blue').astype(int)
    sample.to_csv('sample_data_preprocessed.csv')