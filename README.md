# document-classification

## Purpose
This is the code to interact with a public document classification webservice.

## Prerequisites
In order to interact with the webservice, you will need the following requirements :
```
pandas
```

The *shuffled-full-set-hashed.csv* file must be placed at the root of the project.

## Commands
The following command will launch a test_suite:
```
python3 test_suite.py <number_of_tests>
```
*The numbe of tests is 10 by default*

You can individually test a file content with the following command:
```
python3 test.py <file_content> <expected_label>
```
*The expected_label parameter is optional. If given, the function will return a boolean whether the predicted label matched the expected label.*

## API
Here is an example of how to access the API
```
http://ec2-3-135-82-244.us-east-2.compute.amazonaws.com:5000/classifier/<document_text>
```
