# Install

How to set up and install the project with its dependencies.

## macOS

Simply run the `setup.sh` script. Be sure to have a virtual environment before doing this, to avoid cluttering the system wide Python packages.

## Windows

### Prerequisites

Download and install the latest version of [Python3](https://www.python.org/downloads/).

Install [Git](https://git-scm.com). We're going to use Git Bash later.

Open Git bash and run the following command:

```bash
pip install virtualenv
```

Create a new virtualenv and store it somewhere. Activate it.

### Install project dependencies

Now, head over to [pythinlibs](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) and download the following .whl packages:

* [numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
* [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
* [scikit-image](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)

Install these packages by running the command:

```bash
pip install [name-of-local-whl-file]
```

We can now install the common requirements from the project. Do this by running:

```bash
pip install -r requirements.txt
```

### Build tensorflow from source

We already have tensorflow installed at this point, but this version will not run correctly for us. Instead we need to build Tensorflow for source.

First, uninstall the tensorflow version we have installed:

```bash
pip uinstall tensorflow
```

Install the following pieces of software:

* [SWIG](http://www.swig.org/download.html)
* [CMAKE](https://cmake.org/download/)
* [Visual Studio 2015](https://www.visualstudio.com/downloads/)
* [NVidia CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
* [NVidia CUDNN 5.1](https://developer.nvidia.com/cudnn)

Make sure Cmake is in the PATH. Unpack SWIG and place it somewhere. Install VS 2015 WITH Python build tools (or
something like that). Install CUDA with all drivers and such. Unpack CUDNN and place it somewhere.

Next steps are taken from the [tensorflow guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/README.md). Open up CMD.exe and follow the steps below:

```cmd
C:\> "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
C:\> "C:\path\to\virtualenv\activate.bat"
C:\> git clone https://github.com/tensorflow/tensorflow.git
C:\> cd tensorflow\tensorflow\contrib\cmake
C:\tensorflow\tensorflow\contrib\cmake> mkdir build
C:\tensorflow\tensorflow\contrib\cmake> cd build
C:\tensorflow\tensorflow\contrib\cmake\build>
```

Now we're going to build tensorflow! Paste this entire line into CMD, but adjust paths to correspond to yours:

```cmd
C:\...\build> cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=C:\swig\swigwin-3.0.10\swig.exe ^
-DPYTHON_EXECUTABLE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python35-32\python.exe ^
-DPYTHON_LIBRARIES=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python35-32\libs\python35.lib ^
-Dtensorflow_ENABLE_GPU=ON ^
-DCUDNN_HOME="C:\cudnn\cudnn"
```

This takes a while. After this we build a .whl file that we can install with pip. Do this with:

```cmd
C:\...\build> MSBuild /p:Configuration=Release tf_python_build_pip_package.vcxproj
```
