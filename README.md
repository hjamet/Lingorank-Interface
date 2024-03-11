# Lingorank-Interface

## Description

This project is a simple interface for the Lingorank project. It allows you to visualise the results that can be obtained using the simplification difficulty estimation models described in the paper.

## Installation 🐼

### Requirements 🐨

This project was developed with Python version 3.9.13 and the following tools :
- [Pyenv](https://github.com/pyenv/pyenv-installer) : Python version manager
- [Poetry](https://python-poetry.org/docs/#installation) : Python package manager

Tools that you can install using the following commands:
```bash
> curl https://pyenv.run | bash
> curl -sSL https://install.python-poetry.org | python3 -
```

Note that these tools are not compulsory, but they are strongly recommended to facilitate installation and the management of project dependencies.

### Installation 🐻

#### With Pyenv & Poetry

Once you have installed the above tools, you can install the project using the following commands:
```bash
> pyenv install 3.9.13 # Installs the version of Python used in the project
> pyenv local 3.9.13 # Defines the version of Python used in the project
> poetry install # Installs project dependencies
```

#### Without Pyenv & Poetry

If you don't want to use Pyenv and Poetry, you can install the project using the following commands (*Make sure you have Python 3.9.13 or equivalent installed on your machine*):
```bash
> python3 -m venv .venv # Creates a virtual environment
> source .venv/bin/activate # Activates the virtual environment
> pip install -r requirements.txt # Installs project dependencies
```


## Repository Convention & Architecture 🦥

### Architecture 🦜

The project is divided into several directories:
```
.
├── src # Contains the source code of the project
├── tests # Contains the tests of the project
```

### Convention 🦦

* **[DOCSTRING]** : We use typed google docstrings for all functions and methods. See https://www.python.org/dev/peps/pep-0484/ and https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html . Docstring and notebooks should be written in English.
* **[TEST]** : All files in src should be summarily tested. Ideally, leave a simplistic example of use commented out at the bottom of each file to demonstrate its use. No need to use PyTest absolutely, we're not monsters either :D
* **[GIT]** : We use the the commit convention described here: https://www.conventionalcommits.org/en/v1.0.0/ . You should never work on master, but on a branch named after the feature you are working after opening an issue to let other members know what you are working on so that you can discuss it. When you are done, you can open a pull request to merge your branch into master. We will then review your code and merge it if everything is ok. Issues and pull requests can be written in French.

> Of course, anyone who doesn't follow these rules, arbitrarily written by a tyrannical mind, is subject to judgmental looks, cookie embargoes and denunciatory messages with angry animal emojis.