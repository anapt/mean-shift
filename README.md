
# Mean-shift

[Mean-shift] is a mathematical procedure, adopted in algorithms, designed in the 70's by Fukunaga and Hostetler. The algorithm is used for:

  - Cluster analysis
  - Computer vision
  - Image processing

## Repository

This repository provides a serial implementation of the algorithm in C language, as well as two versions of the parallel equivalent in CUDA, with and without the usage of shared memory. The project was undertaken as part of the "Parallel and distributed systems" course of AUTH university.

A [Gaussian] kernel was used for the weighting function. The code was tested for different data sets and information regarding the execution time and correctness were extracted. In addition, the two versions of the parallel algorithm were tested and compared.

## Dependencies

For the serial algorithm only a compiler is needed (e.g. gcc).

To compile the parallel versions, the standard CUDA toolkit installation instructions for the intended platform should be followed beforehand as described [here].

## Compilation

To compile make sure all necessary packages and dependencies are installed. Then run:

```sh
$ make
```

## Usage

Run the code with the command:
```sh
$ ./meanshift h e N D Pd Pl
```
where:

 1. **h** is the desirable variance
 2. **e** is the min distance, between two points, that is taken into account in computations
 3. **N** is the the number of points
 4. **D** is the number of dimensions of each point
 5. **Pd** is the path of the dataset file
 6. **Pl** is the path of the labels file
 7. **--verbose** | **-v** is an optional flag to enable execution information output
 8. **--output** | **-o** is an optional flag to enable points output in each iteration


**Free Software, Hell Yeah!**

[//]: # (Links)

   [Mean-shift]: <https://en.wikipedia.org/wiki/Mean_shift>
   [Gaussian]: <https://en.wikipedia.org/wiki/Gaussian_function>
   [here]: <https://docs.nvidia.com/cuda/cuda-quick-start-guide/>
