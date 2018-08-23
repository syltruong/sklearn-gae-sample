# Google App Engine - `sklearn` sample

Supporting article: [Deploying Machine Learning has never been so easy](https://towardsdatascience.com/https-towardsdatascience-com-deploying-machine-learning-has-never-been-so-easy-bbdb500a39a)

This is a sample project for deployment of a wrapped Scikit-Learn (`sklearn`) estimator (`TextClassifier`) on an AppEngine instance.

The project assumes a Google Cloud project has been instantiated and the Google Cloud SDK (`gcloud` command-line) has been installed.

__Deployment command__

From the project's root directory: `gcloud app deploy`

__Architecture__

![flow_chart](https://user-images.githubusercontent.com/24707558/43558900-3731f7cc-963e-11e8-8e19-d4ca01bceb9e.png)
