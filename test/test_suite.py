#!/usr/bin/python

import sys
from test import test_prediction
import pandas as pd


def test_suite(n=10):
    df = pd.read_csv("shuffled-full-set-hashed.csv", header=None, names=["Label", "Data"]).dropna(axis=0,
                                                                                                  how='any')

    successes = 0
    for i in range(n):
        print("========= Test " + str(i) + " of " + str(n) + " =========")
        test_row = df.sample(n=1)
        expected = str(test_row['Label'].iloc[0])
        test_content = test_row['Data'].iloc[0]
        if test_prediction(test_content, expected):
            successes += 1
        print("===== Current accuracy = " + str(round(successes/(i+1)*100, 2)) + "% =====\n\n")
    print("Number of tests : " + str(n))
    print("Number of successes : " + str(successes))
    print("Accuracy : " + str(round(successes/n*100, 2)) + "%")


def main():
    if len(sys.argv) > 1:
        test_suite(int(sys.argv[1]))
    else:
        test_suite()

if __name__ == '__main__':
    main()
