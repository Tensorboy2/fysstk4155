# Project 2 for Fysstk 4155 fall 2024
## **by Sigurd Vargdal and Michele Tognoni**
***
### The title for the project:
# Classification and Regression, from linear and logistic regression to neural networks

***

> #### To do:
> 1. First part (*part a*) is to make a simple *regression* optimizer based on stocastic gradient decent. 
>   - The regression method shuld be able to run: 
>       1. *Gradient decent* (with and witout momentum)
>       2. *Stocastic gradient decent* (with and without momentum)
>       3. For both *GD* and *SDG* *AdaGrad should be added (with and without momentum)
>       4. For SGD without momentum *RMSprop* and *Adam* should be added. 
>   - (All this should pretty much be its own python class)
> 2. The second part is about wiriting the neural network (Feed Forward). 
>   - This is mostly coded for arbitrary input sizes and has woriking *GD*.
>       - Add all the gradient methods as possible optimizers
>       - Add also different activiation functions other than sigmoid.
>       - Study also the initialization of the weights and biases.

***

> #### Current status:
> The main function calls the prepare_data function which runs a check if the data exists. If the data does not exist then it runs the get_data to try to download the data through a api call. If the data is found it is then put into a pandas data frame and split into a train and test part and returned in the main function.
> The plan forward is to implement the rest of the analysis tools needed for the project.