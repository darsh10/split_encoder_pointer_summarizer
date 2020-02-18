PyTorch implementation of *[Automatic Fact-guided Sentence Modification](https://arxiv.org/pdf/1909.13838.pdf)* (AAAI 2020).


The code for the Masker is [here](https://github.com/TalSchuster/TokenMasker).


______________________________________________________________________________________________________________________________

Repository cloned and updated from https://github.com/atulkum/pointer_summarizer.



Note:
* It is tested on pytorch 0.4 with python 2.7

______________________________________________________________________________________________________________________________

Training the Model:

<p><code> export PYTHONPATH=`pwd` &&
  python training_ptr_gen/train.py
  </code></p>

______________________________________________________________________________________________________________________________


Dataset:

The dataset for training this model can be found here https://drive.google.com/open?id=1aOMEUksFpZwJDtQcgsrJ0rjC7nO2J1kr.

(Download and edit the config file to the path of the train, val, test and vocab files.)


______________________________________________________________________________________________________________________________

Evaluation:

<p><code>
  export PYTHONPATH=`pwd` &&
  python training_ptr_gen/eval.py _path_of_model_checkpoint
</code></p>

This will generate the corresponding outputs for the desired eval file (specified in the validation path).

______________________________________________________________________________________________________________________________

If you find this repository helpful, please cite our paper:
```
@inproceedings{shah2020automatic,
  title={Automatic Fact-guided Sentence Modification},
  author={Darsh J Shah and Tal Schuster and Regina Barzilay},
  booktitle={Association for the Advancement of Artificial Intelligence ({AAAI})},
  year={2020},
  url={https://arxiv.org/pdf/1909.13838.pdf}
}
```
