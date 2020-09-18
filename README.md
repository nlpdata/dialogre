DialogRE
=====

This repository maintains **DialogRE**, the first human-annotated dialogue-based relation extraction dataset. Please contact dialogre@dataset.org if you have any questions or suggestions.

* Paper: https://arxiv.org/abs/2004.08056
```
@inproceedings{yu2020dialogue,
  title={Dialogue-Based Relation Extraction},
  author={Yu, Dian and Sun, Kai and Cardie, Claire and Yu, Dong},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/2004.08056v1}
}
```

Files in this repository:

* ```license.txt```: the license of DialogRE.
* ```data/{train,dev,test}.json```: the original dataset files (used in the paper). The data format is as follows.
```
[
  [
    [
      dialogue 1 / turn 1,
      dialogue 1 / turn 2,
      ...
    ],
    [
      {
        "x": dialogue 1 / instance 1 / argument 1,
        "y": dialogue 1 / instance 1 / argument 2,
        "x_type": dialogue 1 / instance 1 / argument 1 type,
        "y_type": dialogue 1 / instance 1 / argument 2 type,
        "r": [
          dialogue 1 / instance 1 / relation 1,
          dialogue 1 / instance 1 / relation 2,
          ...
        ],
        "rid": [
          dialogue 1 / instance 1 / relation 1 id,
          dialogue 1 / instance 1 / relation 2 id,
          ...
        ],
        "t": [
          dialogue 1 / instance 1 / relation 1 trigger,
          dialogue 1 / instance 1 / relation 2 trigger,
          ...
        ],
      },
      {
        "x": dialogue 1 / instance 2 / argument 1,
        "y": dialogue 1 / instance 2 / argument 2,
        "x_type": dialogue 1 / instance 2 / argument 1 type,
        "y_type": dialogue 1 / instance 2 / argument 2 type,
        "r": [
          dialogue 1 / instance 2 / relation 1,
          dialogue 1 / instance 2 / relation 2,
          ...
        ],
        "rid": [
          dialogue 1 / instance 2 / relation 1 id,
          dialogue 1 / instance 2 / relation 2 id,
          ...
        ],
        "t": [
          dialogue 1 / instance 2 / relation 1 trigger,
          dialogue 1 / instance 2 / relation 2 trigger,
          ...
        ],
      },
      ...
    ],
  ],
  [
    [
      dialogue 2 / turn 1,
      dialogue 2 / turn 2,
      ...
    ],
    [
      {
        "x": dialogue 2 / instance 1 / argument 1,
        "y": dialogue 2 / instance 1 / argument 2,
        "x_type": dialogue 2 / instance 1 / argument 1 type,
        "y_type": dialogue 2 / instance 1 / argument 2 type,
        "r": [
          dialogue 2 / instance 1 / relation 1,
          dialogue 2 / instance 1 / relation 2,
          ...
        ],
        "rid": [
          dialogue 2 / instance 1 / relation 1 id,
          dialogue 2 / instance 1 / relation 2 id,
          ...
        ],
        "t": [
          dialogue 2 / instance 1 / relation 1 trigger,
          dialogue 2 / instance 1 / relation 2 trigger,
          ...
        ],
      },
      {
        "x": dialogue 2 / instance 2 / argument 1,
        "y": dialogue 2 / instance 2 / argument 2,
        "x_type": dialogue 2 / instance 2 / argument 1 type,
        "y_type": dialogue 2 / instance 2 / argument 2 type,
        "r": [
          dialogue 2 / instance 2 / relation 1,
          dialogue 2 / instance 2 / relation 2,
          ...
        ],
        "rid": [
          dialogue 2 / instance 2 / relation 1 id,
          dialogue 2 / instance 2 / relation 2 id,
          ...
        ],
        "t": [
          dialogue 2 / instance 2 / relation 1 trigger,
          dialogue 2 / instance 2 / relation 2 trigger,
          ...
        ],
      },
      ...
    ],
  ],
  ...
]
```

* ```data_v2/en/data/{train,dev,test}.json```: the updated dataset files with a few annotation errors fixed. The format is the same as the orignal. (**Updated on Aug 2020**)
* ```data_v2/cn/data/{train,dev,test}.json```: a Chinese version of DialogRE. The format is the same as the orignal. Please note that since ground truth argument types do not substantially contribute to the performance according to Section 5.2 of our paper, we no longer annotate argument types when annotating the Chinese version. Instead, all ```"x_type"```s and ```"y_type"```s are left empty. (**Updated on Aug 2020**)
* ```kb/Fandom_triples```: relational triples from [Fandom](https://friends.fandom.com/wiki/Friends_Wiki).
* ```kb/matching_table.txt```: mapping from Fandom relational types to DialogRE relation types.
* ```bert``` folder: a re-implementation of BERT and BERT<sub>S</sub> baselines.
  1. Download and unzip BERT from [here](https://github.com/google-research/bert), and set up the environment variable for BERT by 
  ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
  2. Copy the dataset folder ```data``` (or ```data_v2/{en,cn}/data``` for the updated version) to ```bert/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.
  4. To run and evaluate the BERT baseline, execute the following commands in ```bert```:
  ```
  python run_classifier.py   --task_name bert  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir bert_f1  --gradient_accumulation_steps 2
  rm bert_f1/model_best.pt && cp -r bert_f1 bert_f1c && python run_classifier.py   --task_name bertf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir bert_f1c  --gradient_accumulation_steps 2
  python evaluate.py --f1dev bert_f1/logits_dev.txt --f1test bert_f1/logits_test.txt --f1cdev bert_f1c/logits_dev.txt --f1ctest bert_f1c/logits_test.txt
  ```
  5. To run and evaluate the BERT<sub>S</sub> baseline, execute the following commands in ```bert```:
  ```
  python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 2
  rm berts_f1/model_best.pt && cp -r berts_f1 berts_f1c && python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 2
  python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt
  ```
**Environment**:
  The code has been tested with Python 3.6 and PyTorch 1.0.

**TODO**:

- [x] Release DialogRE
- [x] Release a Chinese version of DialogRE (summer 2020)
- [x] Fix the annotation errors in DialogRE and release an updated English version (summer 2020) 
- [x] Baseline results for the updated version ([here](https://dataset.org/dialogre/))
