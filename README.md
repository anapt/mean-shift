# Mean-shift

[Mean-shift] is a mathematical procedure, adopted in algorithms, designed in the 70's by Fukunaga and Hostetler. The algorithm is used for:

  - Cluster analysis
  - Computer vision
  - Image processing

## Repository

This repository provides a serial implementation of the algorithm in C language, as well as the parallel equivalent in CUDA. The project was undertaken as part of the "Parallel and distributed systems" course of AUTH university.

A [Gaussian] kernel was used for the weighting function. The code was tested for different data sets and information regarding the execution time and correctness were extracted. In addition, two versions of the parallel algorithm were tested and compared, with and without the usage of shared memory respectively.

## Compilation

To compile make sure all necessary packages and dependencies are installed. Then run:

```sh
$ make
```

## Usage

blah blah, arguments needed etc


**Free Software, Hell Yeah!**

[//]: # (Links)

   [Mean-shift]: <https://en.wikipedia.org/wiki/Mean_shift>
   [Gaussian]: <https://en.wikipedia.org/wiki/Gaussian_function>