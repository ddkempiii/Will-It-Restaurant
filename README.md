# "Will It Restaurant?", An Objective Inquiry into Restaurant Ratings

## Introduction

Relied upon for sustenance, entertainment, and social venturing, restaurants play a vital role in any lively city. Yet, not all restaurants excel in their function, 
or do so at varying degrees. Furthermore, the variety of consumer tastes is complex and not at all restricted merely to culinary considerations. Some customer 
demographics can tolerate poorer food quality or flavor, for example, if the type of restaurant ambience (interior decor, music, furniture...) resonates with them, or the customer service is affable and attentive. Or, maybe a large enough consumer base will cherish a certain restaurant because it is close to them while others
find it easy to access. It is unreasonable that a single restaurant could cater to the whole, vast array of consumer preference; hence, the status quo is that a 
typical restaurant will focus on just a certain set of consumer preferences. 

In turn, for the hungry or outgoing consumer, making an effective choice among the vast collection of restaurants in their vicinity can be a daunting
task, without help. Restaurant ratings, e.g. Google, Yelp, etc., have been devised as an aid here. Yet, it can be unclear what exactly they are communicating,
since they are subjective judgments by customers. Indeed, reviews can be read individually to locate some rationale for each customer's restaurant rating, yet 
some reviews are too brief for that or are maybe even left blank. In any case, it is a tedious and potentially unwieldy process to read through every restaurant review and aggregate the ratings into manifold patterns of consumer sentiment.

Thus, the goal of this project is to find some objective basis for restaurant ratings. Using only elementary, measurable data, we investigate to what extent restaurant ratings can be predicted.

## Data

The data pertains to a collection of 3787 restaurants in Philadelphia, PA, and was obtained from Google Maps using Outscraper. The target variable is the Google 
average star rating (a numerical value between 0.0 and 5.0 with one decimal place allowed). Many possible predictor variables are provided by Google Maps, but we 
give special attention to the following: 'reviews' (the total number of customer reviews posted); 'photos_count' (the total number of photos posted by customers); 
'latitude' and 'longitude'; 'range' (the restaurant's price range); 'type' (the type of cuisine); 'site' (the restaurant's website); 'phone' (its phone number);
and 'borough'. There are other possibly helpful variables, e.g. 'other_hours', 'about', 'description', 'popular_times', 'typical_time_spent', which were either returned 
entirely empty by Outscraper or are temporarily ignored here.

We believe that local demographics are an important part of the story behind restaurant ratings. So we also incorporate data from the 2021 US Census administered by
the American Community Survey for the census tracts covering Philadelphia. 

## Preprocessing

We first check that all restaurants are indeed in Philadelphia, and then proceed to locate missing values. All restaurants with missing rating are discarded, and
various columns that are irrelevant or almost entirely empty are also discarded. Not all restaurants in the dataset are operational, nevertheless we retain their 
entries since Google Maps only retains closed restaurants for a short period of time. There are many missing values in the 'site' and 'phone' columns, so these 
columns are changed into Boolean data.

Lastly, the census data in its initial form from ACS presents only whole number values. We must manually calculate the percentages which are desired, and in this
final form, we present the census data for modeling use.


## Model Selection and Results

When we obtain our training set and make plots, the results are very amorphous. Therefore, we are led to attempt k-neighbors regression at the onset. Incorporating
pipelines with basic rescaling from sklearn, we regress with a variety of different predictors. Initially, borough identity is used mainly to track where certain 
models outperform the others in cross-validation, but it becomes the basic criterion for clustering. Price range enables further clustering, where random forest
regression is also found effective. The inclusion of the census predictors sharpens both the random forest and k-neighbors regression models among these clusters. 
Obtaining a preliminary model-cluster pairing across the dataset, we finally train the models on the full training set and apply them to the test set. An RMSE of 0.47
is returned for the test set ratings.

We had some curiosity as to whether the categorical variables alone could locate the core statistical trends in the restaurant data. Indeed, the 'reviews' and
'photos_count' predictors seemed to significantly affect the ratings distribution once they attained certain numeric thresholds. Similarly, it was speculated that
certain areas, not large enough to be boroughs (perhaps just a few adjacent street blocks), may hold high-performing or perhaps drastically underwhelming restaurants.
In an effort to gain hold of such scenarios, a manual form of clustering was also pioneered, called **deconstruction**. 'Deconstruction' should be interpreted as a greedy algorithm that takes a best performing model on a subset of data and uses its statistical predictions to locate a divergent smaller set via decision tree classification. However, it is still in nascent form and needs fine
tuning; its use on the test dataset returns a much higher RMSE of ~0.74.

## Files

### Notebooks:

* __[`data_presentation:cleaning.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_presentation%3Acleaning.ipynb)__ presents, cleans, and validates the data
* __[`data_exploration:plotanalysis.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_exploration%3Aplotanalysis.ipynb)__ performs EDA on the training data and speculates about model candidates
* __[`data-modeling1.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/initial_modeling/data-modeling1.ipynb)__ tests various k-neighbors models
* __[`data-modeling2(price_level1).ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/initial_modeling/data-modeling2(price_level1).ipynb)__ applies the same models to just price level 1 restaurants
* __[`census_data_acquisition.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/census_data_acquisition.ipynb)__ acquires the census data
* __[`census_and_restaurant_data_withEDA.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/census_and_restaurant_data_withEDA.ipynb)__ performs EDA on the training price level 1 restaurants with census predictors
* __[`modeling_with_censusdata.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/initial_modeling/modeling_with_censusdata.ipynb)__ gauges the possible impact of census predictors by using random forest and gradient boost regression, particularly for feature selection
* __[`more_models_withclustering.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_modeling/more_models_withclustering.ipynb)__ tests appropriate random forest and boost models on price level 1 restaurants, providing further evidence for clustering; begins clustering on 'West Philly' while introducing deconstruction
* __[`clustering_westphilly.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_modeling/clustering_westphilly.ipynb)__ finishes the clustering on west philly begun in the previous notebook, making first use of the DecCluster class from __[`cluster_fns.py`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_fns.py)__
* __[`clustering_nephilly.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_modeling/clustering_nephilly.ipynb)__ accomplishes the clustering on NE Philly, also using DecCluster
* __[`model_test.ipynb`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/model_test.ipynb)__ selects and validates the best models for the whole of the data, using the full training set and test set; an assessment of the efficacy of deconstruction is made at the end

### Custom Classes:

* __[`cluster_fns.py`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_fns.py)__ contains the function and class definitions needed to perform deconstruction on a given borough with corresponding data
* __[`tree_helpers.py`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/tree_helpers.py)__ is a file written by Steven Gubkin, Ph.D., on behalf of the Erdős Data Science Boot Camp, which we use here to traverse classification decision trees and return all predictor constraints

### CSV:

* __[`train_data.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/train_data.csv)__ is the training data
* __[`test_data.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/test_data.csv)__ is the test data
* __[`census_tracts.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/census_tracts.csv)__ presents the census tracts covering Philly returned by the US Census Bureau via geocoding
* __[`census_data.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/census_data.csv)__ is the desired census data associated to each tract
* __[`train_data_with_census.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/train_data_with_census.csv)__ is the training data with census tract number attached
* __[`final_train_data.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/data_files/final_train_data.csv)__ is the training data with full census data adjoined
* __[`clusters.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/cluster_modeling/clusters.csv)__ presents the information necessary for using DecCluster to construct the clusters derived on West Philly and NE Philly, with both model and feature specification
* __[`features.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/feature_selection/features.csv)__ is the list of initial features considered
* __[`feature_importances.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/feature_selection/feature_importances.csv)__ gives the random forest feature importances for geospatial, customer-interactive, and census predictors
* __[`feature_importances_w_cuisine_types.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/feature_selection/feature_importances_w_cuisine_types.csv)__ accomplishes the same task but with cuisine type included as well as a predictor
* __[`final_features.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/feature_selection/final_features.csv)__ selects the most important features from __[`feature_importances.csv`](https://github.com/ddkempiii/Will-It-Restaurant/blob/main/feature_selection/feature_importances.csv)__
