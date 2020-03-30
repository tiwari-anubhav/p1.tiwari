### FROM CMD or TERMINAL
 Navigate to the folder where the project is located. Run using the following command.
 >>>> python </path>/main.py
 
### The program then prompts the user to enter his choice for running the algorithms in following three scenarios:
 1.) KNN alone
 2.) SVM alone
 3.) KNN and SVM concurrently
 
** User can enter his/her value of k for KNN classification and gamma factor for SVM algorithm.
Source Code Contains four scripts or .py files:
1.) Classification.py -  Containing the dataset and the function to preprocess the dataset and split it into training and test dataset.
2.) KNNClassification.py - Contains KNN classifier function
3.) SVMClassification.py - Contains SVM classifier function
4.) main.py - Controls the execution of all the scripts using the multi-threads.

## RESULTS (Running time and Accuracy of KNN (k=5) is better than SVM)
User Level Information:
1) Running Time

System-level information:
1) CPU usage
2) Memory usage
3) Hard drive usage
4) RSS: Resident set size: This is non-swapped physical memory that a process had used
5) VMS: Virtual memory size: This is the total virtual memory used by the process
6) Number of page faults

The two classifier are compared for their efficiency and the results are shown with the help of a call graph using the following commands.
        python -m cProfile -o program.profile test.py
        pyprof2calltree -i program.profile -o program.calltree
        qcachegrind program.calltree


### NOTES on data encoding

Dataset used is iris dataset available in sklearn.
Only two features, the petal length and width have been used for this analysis.
View the first 5 rows of the data
   petal length (cm)  petal width (cm)
0                1.4               0.2
1                1.4               0.2
2                1.3               0.2
3                1.5               0.2
4                1.4               0.2

The unique labels in this data are [0 1 2]

After standardizing our features, the first 5 rows of our data now look like this:

   petal length (cm)  petal width (cm)
0          -0.182950         -0.291459
1           0.930661          0.737219
2           1.042022          1.637313
3           0.652258          0.351465
4           1.097702          0.737219
