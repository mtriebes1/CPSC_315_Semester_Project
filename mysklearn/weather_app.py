from flask import Flask, jsonify, request

import os
import pickle

# make a flask app
app = Flask(__name__)

# we need to add two routes (funcitons that handle requests)
# One for the homepage
@app.route("/", methods=["GET"])
def index():
    # Return content and a status code
    return "<h1>Welcome to my App</h1>", 200
# One for the API /predict endpoint
@app.route("/predict", methods=["GET"])
def predict():
    minTemp = request.args.get("MinTemp", "")
    maxTemp = request.args.get("MaxTemp", "")
    windGustDir = request.args.get("WindGustDir", "")
    humidity9am = request.args.get("Humidity9am","")
    humidity3pm = request.args.get("Humidity3pm","")
    pressure9am = request.args.get("Pressure9am","")
    pressure3pm = request.args.get("Pressure3pm","")

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response
    prediction = predict_interviews_well([minTemp, maxTemp, windGustDir, humidity9am, humidity3pm, pressure9am, pressure3pm])
    # If anyting goes wrong, this function returns None
    if prediction is not None:
        result = {"prediction:", prediction}
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

def predict_weather(instance):
    # 1. We need a tree (and its header) to make a prediction
    #   - we need to save a trained model (fit()) to a file so we
    #     can load that file into memory in another python process
    #   - Import pickle and load the header and the tree 
    #     into memory for use in part 2 
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    
    # 2. use the tree to make a prediction
    try: 
        return tdidt_predict(header, instance, tree)
    except:
        return None

def tdidt_predict(header, instance, tree):
    info_type = tree[0]
    if info_type == "Attribute":
        instance_attribute_value = instance[header.index(tree[1])]
        # Now I need to find which branch to follow recursively
        for kk in range(2, len(tree)):
            value_list = tree[kk]
            if value_list[1] == instance_attribute_value:
                # We have a match - recurse through the rest of tree
                return tdidt_predict(header, instance, value_list[2])
    else: # "Leaf"
        return tree[1]

if __name__ == "__main__":
    # Deployment Notes
    #
    # Two main categories of how to deploy:
    #   1. Host your own server
    #   2. Use a cloud provider: there are lots of options - AWS, Heroku, Azure, ...
    # We are going to use Heroku (Backend as a Service BaaS) 
    # There are lots of ways to deploy a Flask app to Heroku
    #   1. Deploy the app directly as a web app running on the ubuntu "stack" (e.g Procfile and requirements.txt)
    #   2. deploy the app as a Docker container running on the container "stack" (e.g. Dockerfile)
    #       2.A. Build the Docker image locally and push it to a container registery (e.g. Heroku)
    #       **2.B.** Define a file called heroku.yml and push our source code to Heroku's git repo
    #                and Heroku will build the docker image for us
    #       2.C. Define a main.yml and push our source code to Github, where a Github Action builds
    #            the image and pushes it to the Heroku registery
    #
    # We will be using method 2.B.

    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # By default, flask runs on port 5000