# pWord2Vec
This is a C++ implementation of word2vec that is optimized on Intel CPUs, particularly, Intel Xeon and Xeon Phi (Knights Landing) processors. It supports the "HogBatch" parallel SGD as described in "[Parallelizing Word2vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661)". A short NIPS workshop version can be found [here](https://arxiv.org/abs/1611.06172). It also uses data parallelism to distribute the computation via MPI over a CPU cluster. The code is developed based on the [original word2vec](https://code.google.com/archive/p/word2vec/) implementation from Google.

##License
All source code files in the package are under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

##Prerequisites
The code is developed and tested on UNIX-based systems with the following software dependencies:

- [Intel Compiler](https://software.intel.com/en-us/qualify-for-free-software) (The code is optimized on Intel CPUs)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
- MPI library, with multi-threading support (Intel MPI, MPICH2 or MVAPICH2 for distributed word2vec only)
- [HyperWords](https://bitbucket.org/omerlevy/hyperwords) (for model accuracy evaluation)
- Numactl package (for multi-socket NUMA systems)

##Environment Setup
* Install Intel C++ development environment (i.e., Intel compiler, OpenMP, MKL "16.0.0 or higher" and iMPI. [free copies](https://software.intel.com/en-us/qualify-for-free-software) are available for some users)
* Enable Intel C++ development environment
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
source /opt/intel/impi/latest/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
```
* Install numactl package
```
sudo yum install numactl (on RedHat/Centos)
sudo apt-get install numactl (on Ubuntu)
```

##Quick Start
1. Download the code: ```git clone https://github.com/IntelLabs/pWord2Vec```
2. Run .\install.sh to build the package (e.g., it downloads hyperwords and compiles the source code.) Note that this installation will try to produce two binaries: pWord2Vec and pWord2Vec_mpi. If you are only interested in the non-mpi version of w2v, you don't need to set up mpi and the compilation will fail on building pWord2Vec_mpi of course. But you can still use the non-mpi binary for the rest of single machine demos.
3. Download the data: ```cd data; .\getText8.sh or .\getBillion.sh```
4. Run the demo script: ```cd sandbox; ./run_single_text8.sh (for single machine demo) or ./run_mpi_text8.sh (for distributed w2v demo)```
5. Run the code on the 1-billion-word-benchmark: ```cd billion; ./run_single.sh (for single machine w2v) or ./run_mpi.sh (for distributed w2v) (please set ncores=number of logical cores of your machine)```
6. Evaluate the models: ```cd sandbox; ./eval.sh or cd billion; ./eval.sh```

##Reference
1. [Parallelizing Word2Vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661), arXiv, 2016.
2. [Parallelizing Word2Vec in Multi-Core and Many-Core Architectures](https://arxiv.org/abs/1611.06172), in NIPS workshop on Efficient Methods for Deep Neural Networks, Dec. 2016.

For questions and bug reports, please email shihao.ji@intel.com
