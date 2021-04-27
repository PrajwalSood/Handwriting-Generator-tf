# Requirements

This project was implmented using python 3.6.8  
the librairies and their respective version are located in the requirements.txt  

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt

# Training the models

python3 train_prediction.py  
python3 train_synthesis.py

Both those files have configuration files for the model to train. Like the number of steps and epochs.


# Visualisaing the results

After training the models you can instantiate the class of your model:  

#### Unconditional generation
from models import HandWritingPrediction

generation = HandWritingPrediction()  
mysentence = generation.infer(seed=0, inf_type=None, weights_path=None, reload=False)  

To plot the results :

from utils import plot_stroke  
plot_stroke(mysentence)

#### Conditional
from models import HandWritingSynthesis  
from data import DataSynthesis  

generation = HandWritingSynthesis()  
D = DataSynthesis()  

sentence = D.prepare_text("Generate this sentence")
mysentence, _, _, _ = generation.infer(sentence, weights_path=None, reload=False, seed=None)  

To plot the results :

from utils import plot_stroke  
plot_stroke(mysentence)
