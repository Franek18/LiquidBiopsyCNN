### Installation
First create a python3 virtual environment in the repository folder
```commandline
python3 -m venv venv
```
activate the environment with commmand
```commandline
source venv/bin/activate
```
and install requirements
```commandline
pip install -r requirements.txt
```

### Running
The main file is ```main.py```. It has three arguments which can be set during running.
These arguments are:
- model - type of model to train, by default it is VanillaCNN
- data - type of data's representation, default is standard (267x531)
- test - type of split of the data, by default it is hospital split

To run a training type
```commandline
python main.py --model Resnet --data standard --test hospital
```