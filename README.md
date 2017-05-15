# Rorschach :: Code

[![Build Status](https://travis-ci.com/OptimusCrime/master-thesis-code.svg?token=JmzjtQYirFw9etqSW57N&branch=master)](https://travis-ci.com/OptimusCrime/master-thesis-code)

Code for the master thesis for Thomas Gautvedt.


## Install

We are using Docker to make the installation of the system easy and platform independent.

## Requirements

- [Docker](https://www.docker.com/community-edition)
- [Docker compose](https://docs.docker.com/compose/install/)

*Note that Docker compose is included in the Docker Toolbox, so if you install that you do not need to install the standalone version.*

We supply a Makefile to make it easier to use this setup. UNIX based systems should have Make installed by default. Windows systems need to install this manually. It is also possible to use this setup without Make, but you will have to type out some of the tasks manualy instead of relying on the provided shortcuts.

## Setup

To start the system for the first time, type

```
make
```

This is a shortcut for the following commands

```
make build
make up
make prepare
make run
```

To stop the machine type:

```
make stop
```

If you already have built and prepared the system you can start it up again with

```
make start
```

To run the system, type

```
make run
```

If you want to SSH into the container, you can do this with

```
make bash
```

There are a few more shortcuts in the Makefile. Browse the file to see these.

If you are unable to use the `make` commands on your system is the best alternative to run each consecutive command found in the Makefile manually. The file and its content should be pretty self-explanatory.

### The `prepare` command

This command runs various sub commands to set up the system correctly:

- Install all the requirements in requirements.txt.
- Downloads wordlists and fonts from [resource repository](https://github.com/OptimusCrime/master-thesis-resources).
- Create a new config-file with a default setup.

## Config file explanation

Foobar
