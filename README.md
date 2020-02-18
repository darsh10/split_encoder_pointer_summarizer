PyTorch implementation of *[Automatic Fact-guided Sentence Modification](https://arxiv.org/pdf/1909.13838.pdf)*


______________________________________________________________________________________________________________________________

Repository cloned and updated from https://github.com/atulkum/pointer_summarizer.



Note:
* It is tested on pytorch 0.4 with python 2.7

______________________________________________________________________________________________________________________________

Training the Model:

`export PYTHONPATH=`pwd`

python training_ptr_gen/train.py`


______________________________________________________________________________________________________________________________


Dataset:

The dataset for training this model can be found here https://drive.google.com/open?id=1aOMEUksFpZwJDtQcgsrJ0rjC7nO2J1kr.

(Download and edit the config file to the path of the train, val, test and vocab files.)


______________________________________________________________________________________________________________________________

Evaluation:

`export PYTHONPATH=`pwd`

python training_ptr_gen/eval.py _path_of_model_checkpoint`
