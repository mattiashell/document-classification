import requests
import pandas as pd

def test_prediction(test_content, expected):



    address = 'http://ec2-3-135-82-244.us-east-2.compute.amazonaws.com'
    port = 5000

    response = requests.get(address + ':' + str(port) + '/classifier/' + test_content)
    predicted = str(response.text.replace('\"', '').replace('\n', ''))



    print("Expected :\t" + expected)
    print("Predicted :\t" + predicted)

    return expected==predicted

