# pWord2Vec
This is a C++ implementation of word2vec that is optimized on Intel Xeon and Xeon Phi (Knights Landing) processors. It supports the "HogBatch" parallel SGD as described in "[Parallelizing Word2vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661)". A short NIPS workshop version can be found [here](https://arxiv.org/abs/1611.06172). In addition, it uses data parallelism to distribute the computation via MPI over a CPU cluster. The code is based on the [original word2vec](https://code.google.com/archive/p/word2vec/) implementation from Mikolov et al.

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
2. Compile the code: ```make clean all```
3. Download the data: ```cd data; .\getsmall.sh or .\get1billion.sh```
4. Run the demo script: ```cd sandbox; ./example_blackout.sh```
5. Run the code on the 1-billion-word-benchmark: ```cd billion; ./run_64k.sh or ./run_800k.sh or ./run_1m.sh (please set the ncores=number of physical cores of your machine)```

##Reference
1. [Parallelizing Word2Vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661), arXiv, 2016.
2. [Parallelizing Word2Vec in Multi-Core and Many-Core Architectures](https://arxiv.org/abs/1611.06172), in NIPS workshop on Efficient Methods for Deep Neural Networks, Dec. 2016.

For questions and bug reports, please email shihao.ji@intel.com
