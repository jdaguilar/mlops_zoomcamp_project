## Deploying a model as a web-service

* Creating a virtual environment with Pipenv
* Creating a script for prediction
* Putting the script into a Flask app
* Packaging the app to Docker


```bash
docker build -t churn-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696  churn-prediction-service:v1
```
