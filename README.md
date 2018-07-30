# Food Classifier

This classifier takes the name of a dish and then automatically categorizes it into one of many food categories. It is used as an API as part of a Kubernetes framework for an app that is based on reviewing dishes from restaurants. The purpose of this is to allow users to simply put in the name of a dish (if not already in the database) and then adding the dish automatically into the database for a swift user-experience, as compared to forcing either users or restaurants to add their dishes into respective categoriest themselves.

## Getting Started

Just download or clone the repository, and run it using Python 2.7 to get the server running. You can communicate with it using Postman or by adding an HTML to the code.

### Prerequisites

The following libraries must be installed
* Numpy/Pandas
* Scikit Learn
* NLTK 
* Flask


### Installing

You can dockerize the following code simply by running:

```
docker build <filename> .
```
Or you can pull the already built docker image by the following command:
```
docker pull asim29/dish_classifier
```
Then run the following command to run the docker image

```
docker run asim29/dish_classifier
```

Or, alternatively, you can first list the images, and copy the image ID 
```
docker image ls
docker run <image id>
```
This will get the server up and running. 

## Running the code

Send a POST requestion to 'localhost:5000/classify' with the name of the dish, and in the request send a JSON object of the form
{"name": "dishname", "category": "priordishcategory", "ID":1234}
Note: The category can be left empty, or it can be named "UR" which stands for Unrecognized. And the ID can be any number.

Please comment out line# 105 when running random tests, as that would add whatever dish you add into the working database of the application.

## Built With

* [Flask](http://flask.pocoo.org/) - The web framework used

## Contributing

Any and all criticism is welcome, just submit an issue, or send an email.

## Acknowledgments

* Udacity - [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) for providing necessary skills to be able to make this classifier
