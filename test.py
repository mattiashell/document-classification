import requests
import sys
import pandas as pd

def test_prediction(test_content, expected=None):


    address = 'http://ec2-3-135-82-244.us-east-2.compute.amazonaws.com'
    port = 5000

    response = requests.get(address + ':' + str(port) + '/classifier/' + test_content)
    predicted = str(response.text.replace('\"', '').replace('\n', ''))




    print("Predicted :\t" + predicted)
    if expected:
        print("Expected :\t" + expected)
        return expected==predicted
    else:
        return predicted

def main():
    if len(sys.argv) == 2:
        return test_prediction(sys.argv[1])
    elif len(sys.argv) == 3:
        return test_prediction(sys.argv[1], sys.argv[2])
    else:
        return test_prediction()

if __name__ == '__main__':
    main()
