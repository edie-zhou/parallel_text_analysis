# Hardware Accelerated Pattern Matching

Rish Bhatnagar (rb42554), Jacob Grimm (jag9497), Josh Kall (jsk2544), Edie Zhou (ez3437)

Spring 2020 final term project for EE 361C Multicore Computing.

Our program uses a parallel implementation of the Boyer-Moore-Horspool Algorithm
to find string patterns in large text files.

Acceptable input to the function is as follows:

Steps to Compile the Program:

There is a make file included in the zip folder which should compile the code correctly.
The executable is named match.exe, and the executable takes in parameters.
The format to execute the program is as follows:
	./match.exe {search_pattern} {text file path}


search_pattern is the target expression you want to find.
text file path must be relative to the directory that the executable is in.

Example of commmand prompt:

./match.exe him sample-texts/BibleKJV.txt  //searching for "him" in The Bible

If running on TACC, use the batch file in the zip file by typing in the following commands:

```{bash}
$ module load cuda
$ module load gcc
$ sbatch match-batch
```

Please modify the batch script to run different patterns and files on a cluster
