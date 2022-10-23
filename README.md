# Introduction to Machine Learning

There is strong demand for machine learning (ML) skills and expertise to solve challenging business problems both globally and locally in KSA. This course will help learners build capacity in core ML tools and methods and enable them to develop their own machine learning applications. This course covers the basic theory behind ML algorithms but the majority of the focus is on hands-on examples using [Scikit Learn](https://scikit-learn.org/stable/index.html).

## Learning Objectives

The primary learning objective of this course is to provide students with practical, hands-on experience with state-of-the-art machine learning and deep learning tools that are widely used in industry.

The first part of this course covers chapters 1-10 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). The following topics will be discussed.

* The Machine Learning Landscape 
* End-to-End Machine Learning Projects 
* Classification 
* Training Models 
* Support Vector Machines (SVMs) 
* Decision Trees 
* Ensemble Methods and Random Forests 
* Dimensionality Reduction 
* Unsupervised Learning Techniques 
* Introduction to Artificial Neural Networks 

## Lessons

### Day 1: [Introduction to Machine Learning, Part I](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/ESpKKIFbCsVIt06sWnhs7RcBniV7RQAUs2jhOwEAenOm4w?e=FWmjvp)

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-1.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-1.ipynb)

* The morning session will focus on the theory behind linear models for solving basic classification and regression problems by covering chapters 1-4 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  
* The afternoon session will focus on applying the techniques learned in the morning session using [Scikit Learn](https://scikit-learn.org/stable/index.html), followed by a short assessment on the Kaggle data science competition platform.

### Day 2: [Introduction to Machine Learning, Part II](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EYVl3sggch1HqEKHZO8O9t4BpXwFB3NCMCM0tLue6H0T8Q?e=z2VLrt)

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-2.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-2.ipynb)

* Consolidation of previous days content via Q/A and live coding demonstrations.  
* The morning session will focus on the theory behind non-linear models for solving basic classification and regression problems by covering chapters 5-7 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  
* The afternoon session will focus on applying the techniques learned in the morning session using [Scikit Learn](https://scikit-learn.org/stable/index.html), followed by a short assessment on the Kaggle data science competition platform.

### Day 3: [Introduction to Machine Learning, Part III](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EeueWHxeVMxKjkth_Qk9f0UBfJhcRRqVMxyXXKJkxC53oA?e=wHk5xD)

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-3.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-machine-learning/blob/main/notebooks/introduction-to-sklearn-part-3.ipynb)

* Consolidation of previous days content via Q/A and live coding demonstrations.  
* The morning session will focus on the theory behind dimensionality reduction and unsupervised learning techniques as well as introducing the theory behind neural networks by covering chapters 8-10 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  
* The afternoon session will focus on applying the techniques learned in the morning session using [Scikit Learn](https://scikit-learn.org/stable/index.html), followed by a short assessment using the Kaggle data science competition platform.

### Day 4: Introduction to Machine Learning, Part IV

* Consolidation of previous days content via Q/A and live coding demonstrations.  
* The morning session will focus on applying the techniques learned over the previous days using Scikit-Learn.
* The afternoon session will allow time for a final assessment as well as additional time for learners to complete any of the previous assessments.

## Assessment

Student performance on the course will be assessed through participation in a Kaggle classroom competition. 

# Repository Organization

Repository organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Name all files to reflect their content or function.

## Building the Conda environment

After adding any necessary dependencies that should be downloaded via `conda` to the 
`environment.yml` file and any dependencies that should be downloaded via `pip` to the 
`requirements.txt` file you create the Conda environment in a sub-directory `./env`of your project 
directory by running the following commands.

```bash
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```

Note that the `ENV_PREFIX` directory is *not* under version control as it can always be re-created as 
necessary.

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment, activate the Conda environment, and build 
JupyterLab with any additional extensions. The script should be run from the project root directory 
as follows. 

```bash
./bin/create-conda-env.sh
```

### Ibex

The most efficient way to build Conda environments on Ibex is to launch the environment creation script 
as a job on the debug partition via Slurm. For your convenience a Slurm job script 
`./bin/create-conda-env.sbatch` is included. The script should be run from the project root directory 
as follows.

```bash
sbatch ./bin/create-conda-env.sbatch
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
$ mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

## Using Docker

In order to build Docker images for your project and run containers with GPU acceleration you will 
need to install 
[Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/), 
[Docker Compose](https://docs.docker.com/compose/install/) and the 
[NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.