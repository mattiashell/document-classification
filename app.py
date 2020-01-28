#!/usr/bin/env python3


from flask import Flask
from flask_restful import Api, Resource, reqparse
from predict import predict

app = Flask(__name__)
api = Api(app)

class Classifier(Resource):
    def get(self, data):
        return predict(data), 200

api.add_resource(Classifier, "/classifier/<string:data>")
app.run(host="0.0.0.0", debug=True)
