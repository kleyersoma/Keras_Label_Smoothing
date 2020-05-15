# Keras Label Smoothing for Supervised Learning

### What is label smoothing and why would it be used?

The model should not become too confident in its predictions, this can be avoided by applying label smoothing, 
this technique can lessen the confidence of the model and prevent it from descending into deep crevices of the loss landscape where 
overfitting occurs. Label Smoothing is form of regularization.

#### There a two methods to implement Label Smoothing:

* Label smoothing by explicitly updating your labels list.
* Label smoothing by using the loss function.

Regularization methods are used to help combat overfitting and help our model generalize. Examples of regularization methods include:

* Dropout.
* L1 and L2 weight decay.
* Data Augmentation.
* Synthetic Data.

However, there is another regularization technique, it is : 
***Turns “hard” class label assignments to “soft” label assignments.*** 

This  new technique operates directly on the labels themselves. 
***Is dead simple to implement and it can lead to a model that generalizes better.***
