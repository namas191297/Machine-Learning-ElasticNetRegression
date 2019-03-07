Elastic-Net regression can be defined as regularization method that combines both, the ridge regression penalty and
the LASSO penalty which are both also known as L2 norm and L1 norm respectively. It is used generally when there
are tons of parameters that exist in a model and extremely useful when parameters are correlated. It is used in
situations of multicollinearity.


Problem : As Elastic-Net is generally implemented on datasets containing tens of thousands of parameters. Instead of a 
a given problem, let us assume and generate random values to be used in the model. These values will be divided into 
training and testing data and then we will compare the differences in the co-efficients and errors between
Ridge, LASSO and Elastic-Net.