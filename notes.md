## Notes from Google course
### Introductory theory
#### Models and data
Machine learning: we try to predict a *label* from *features*. Features describe / represent data; it's information on data. We call **model** the thing which is going to do the prediction, and which we will train to do so. Models define the relationship between the features and the label, and again, it is shaped with training.
- **Regression**: a model which predicts continuous values, e.g. numbers (money).
- **Classification**: a model which predicts discrete values, e.g. labels (animal species).

A simple example of a linear model is the following:
<center>y = wx + b</center>
in which
- *y* is the value you are trying to predict (also called *desired output*)
- *w* is the slope
- *x* is the value of the input feature
- *b* is the bias, sometimes called w<sub>0</sub>
- we may have subscripts as w<sub>i</sub> and x<sub>i</sub> because we may be in more than one dimension

We estimate the accuracy of the prediction model with a **loss function**.

#### Loss functions
- A simple loss function can measure the distance from the line (predicted outcome) of the outlier examples.
- Another simple loss function is the **squared error** (L2), i.e. the square of the difference between the true value and the predicted value.
	-  Usually we prefer to compute loss over the whole dataset, that's why we use the *sum of squared errors*, i.e. summation over all the elements in the dataset of (true value - predicted value)<sup>2</sup>.
- Another loss function is the **mean square error** (MSE), i.e. L2 averaged on the whole dataset. It's 1/N * SUM (desired-predicted)<sup>2</sup>.

**Training** a model means finding good values for the weights, such that the loss function is minimized across all examples.

- To minimize the loss function, inversely follow the direction of the gradient, *the derivative of the loss function with respect to the model parameters*.
- To put it simply, feed the model into the data, obtain results, compute loss, compute gradient, update model parameters, restart. It's an iterative process (**Gradient descent method**).
	- The gradient tells the derivative (slope) of the curve; it has a magnitude and a direction, and always points in the direction of the steepest increase in the loss function.
	- *Stochastic Gradient Descent*: compute gradient over one example at a time. Results show that a good solution is found after a reasonable amount of steps (n way smaller than dataset size)
	- *Mini-Batch Gradient Descent*: compute gradient over small batches of data (compared to dataset size)
		- We call a **batch** the total number of examples used to calculate the gradient on a single interation.
- Toggle the **learning rate** or *step size* parameter to adjust learning speed
	- Small learning rate means little steps and slow learning
	- Large learning rate means large steps and quick learning
	- Both have pros and cons.  
- Distinguish convex problems from non-convex problems.
	- COnvex problems have only one minimum.
	- We need an *initial guess* especially for non-convex function (i.e. a "good" point to start the learning process from).

When it is not possible to update the model any more, or the loss is small enough, we say that the model has *converged*.

### Notes on `pandas`
`pandas` is a column-oriented data analysis API. Its primary structures are
- `DataFrame`, which can be imagined as a relational table with rows and columns
- `Series`, which is a single column.
Create a series as
````python
import pandas as pd
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
````
and import them in a `DataFrame`
````python
pd.DataFrame({ 'City name': city_names, 'Population': population })
````
In this way you can access
- `DataFrame`s: `cities = pd.DataFrame(pd.DataFrame({ 'City name': city_names, 'Population': population })`
-  `Series`: `cities['City name']`
Interesting: possibility to manipulate data. It is possible to do
- basic arithmetic operations
-  apply NumPy operators
-  use `Series.apply` followed by a lambda function for complex single-column transformations

Know that both `DataFrame`s and `Series` define an `index` property that assigns an identifier value to each `Series` item or `DataFrame` row. If an `index` does not match any row or column present in the (original) `DataFrame`, the corresponding entry will be added with NaN value. 

### TensorFlow notes
In this exercise, we will use 1990 Census data about housing in California.
- We want to predict `median_house_value`, which will be our **label** or **target**
- To predict labels, we use *features*. Features - and data types - can be either Categorical or Numerical.
- We indicate a feature's data type through the construct **feature column**. Feature columns do NOT store the data itself, but only its description.
