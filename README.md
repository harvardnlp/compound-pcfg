# Compound Probabilistic Context-Free Grammars
Code for the paper:  
[Compound Probabilistic Context-Free Grammars for Grammar Induction](https://arxiv.org/abs/1906.10225)  
Yoon Kim, Chris Dyer, Alexander Rush  
ACL 2019  

The preprocessed datasets, trained models, and the datasets parsed with the trained models can be found [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view?usp=sharing).

## Dependencies
The code was tested in `python 3.6` and `pytorch 1.0`. We also require the `nltk` package if creating
the processed data from the raw PTB dataset.

## Data  
The processed version of PTB can be downloaded at the above link. This contains the train/validation/test sets, as well as the vocabulary used (`ptb.dict`). If you want to create this from scratch, you can run
```
python process_ptb.py --ptb_path PATH-TO-PTB/parsed/mrg/wsj --output_path DATA-FOLDER
```
where `PATH-TO-PTB` is the location of your PTB corpus and `OUTPUT-FOLDER` is where the processed trees are saved. This will create `ptb-train.txt`, `ptb-valid.txt`, `ptb-test.txt` in `DATA-FOLDER`.

Now run the preprocessing script
```
python preprocess.py --trainfile data/ptb-train.txt --valfile data/ptb-valid.txt 
--testfile data/ptb-test.txt --outputfile data/ptb --vocabsize 10000 --lowercase 1 --replace_num 1
```
See `preprocess.py` for more options (e.g. batch size).
Running this will save the following files in the `data/` folder: `ptb-train.pkl`, `ptb-val.pkl`, `ptb-test.pkl`, `ptb.dict`. Here `ptb.dict` is the word-idx mapping, and you can change the output folder/name by changing the argument to `--outputfile`.

## Training
To train the compound PCFG, run
```
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl 
--save_path compound-pcfg.pt --gpu 0
```
where `--save_path` is where you want to save the model, and `--gpu 0` is for using the first GPU in the cluster (the mapping from PyTorch GPU index to your cluster's GPU index may vary).

To train the neural PCFG that does not use continuous latent variables, run 
```
python train.py --z_dim 0 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl 
--save_path neural-pcfg.pt --gpu 0
```
Training will take 2-4 days depending on your setup.

## Evaluation
To evaluate the trained model on the test set, run
```
python eval.py --model_file compound-pcfg.pt --data_file data/ptb-test.txt 
--out_file pred-parse.txt --gold_out_file gold-parse.txt --gpu 0
```
where `--out_file` is where you want to output the predicted parses. This will calculate the F1 scores between the predicted trees and the gold trees in `ptb-test.txt`.

To parse a new set of sentences, run
```
python parse.py --model_file compound-pcfg.pt --data_file sents-to-be-parsed.txt 
--out_file pred-parse.txt --gpu 0
```
Note that `sents-to-be-parsed.txt` should have one sentence per line, and be preprocessed in a way that roughly matches the processing in `process_ptb.py` (e.g. no punctuation).

To just evaluate F1 given the trees, run (for example)
```
python compare_trees.py --tree1 data/parsed-data/ptb-test-gold-filtered.txt
--tree2 data/parsed-data/ptb-test-compound-pcfg.txt
```

### Note regarding F1 calculation
To be comparable to the numbers reported in PRPN/Ordered Neurons papers, 
we use the original sentence F1 evaluation code based on 
L83-89 of the [PRPN repo](https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py).
This has quirky behavior in corner cases where the gold tree is over a sentence of length > 2 
but only has the sentence-level trivial span. 
In this case the sentence F1 for that example could be potentially nonzero according to the code. 
Corpus F1 does not have this issue.

## Trained models
We provide the best neural/compound PCFG models, under the `data/trained-model` folder. These can be used for `eval.py` or `parse.py`.

## Parsed Datasets
We also provide parsed train/val/test sets for the best run of each model for further analysis and
RNNG training. These can be found under the `data/parsed-data` folder when you download the processed datasets from above:  
- `ptb-{train/valid/test}-gold-filtered.txt`: Gold trees where length 1 sentences have been filtered out.  
- `ptb-{train/valid/test}-{neural-pcfg/compound-pcfg/prpn/on}.txt`: Predicted trees from the best run of different models. For example `ptb-test-compound-pcfg.txt` is the test set parsed with the compound PCFG. 
- `ptb-test-{neural-pcfg/compound-pcfg/prpn/on}-{rnng/urnng}.txt`: Predicted trees for the test only for RNNG and URNNG (i.e. train on induced trees with RNNG, then fine-tune with URNNG objective). For example `ptb-test-compound-pcfg-urnng.txt` contains the predicted trees from an
an RNNG trained on compound PCFG trees then fine-tuned with the URNNG objective.

### Results
Here are the sentence-level F1 numbers on the PTB test set for the models that performed best on the validation set . 
`F1 with Induced URNNG` indicates training an RNNG on the induced trees and then fine-tuning with the URNNG objective (see below).
This gave improvements across the board.

| Model          | F1          | F1 with Induced URNNG |
|----------------|-------------|-----------------------|
| PRPN           | 47.9        | 51.5                  |
| Ordered Neurons| 50.0        | 55.1                  |
| Neural PCFG    | 52.6        | 58.7                  |
| Compound PCFG  | 60.1        | 66.9                  |


## Training Recurrent Neural Network Grammars (RNNG) on Induced Trees
Training the RNNG on induced trees and fine-tuning with the Unsupervised RNNG uses code from
[Unsupervised Recurrent Neural Network Grammars](https://github.com/harvardnlp/urnng). The below commands should be
run from the `urnng` folder.

First preprocess the training set with induced trees, for example with the compound PCFG:
```
python preprocess.py --batchsize 16 --vocabfile data/ptb.dict --lowercase 1 --replace_num 1
--trainfile data/parsed-data/ptb-train-compound-pcfg.txt 
--valfile data/parsed-data/ptb-valid-compound-pcfg.txt 
--testfile data/parsed-data/ptb-test-compound-pcfg.txt 
--outputfile data/ptb-comp-pcfg
```
Note the use of `--vocabfile` to use the same vocabulary as the one used in the above experiments.

Then use the above files to train an RNNG (and fine-tune with URNNG) using instructions from the
URNNG folder, e.g. 

```
python train.py --train_file /compound-pcfg/data/ptb-comp-pcfg-train.pkl 
--val_file /compound-pcfg/data/ptb-comp-pcfg-val.pkl --save_path compound-pcfg-rnng.pt 
--mode supervised --train_q_epochs 18 --count_eos_ppl 1 --gpu 0
```
For this version of PTB we count the `</s>` token in PPL calculations, hence 
the use of `--count_eos_ppl 1`. Note that this only affects evaluation and not training.

For fine-tuning:
```
python train.py --train_from compound-pcfg-rnng.pt --save_path compound-pcfg-urnng.pt
--train_file /compound-pcfg/data/ptb-comp-pcfg-train.pkl 
--val_file /compound-pcfg/data/ptb-comp-pcfg-val.pkl
--mode unsupervised --train_q_epochs 10 --epochs 10 --count_eos_ppl 1 --lr 0.1 --gpu 0 --kl_warmup 0
```

For evaluation:
```
python eval_ppl.py --model_file compound-pcfg-urnng.pt --samples 1000 --is_temp 2 --gpu 0
--test_file /compound-pcfg/data/ptb-test.pkl --count_eos_ppl 1
```

For parsing F1:
```
python parse.py --model_file compound-pcfg-urnng.pt --data_file /compound-pcfg/data/ptb-test.txt 
--out_file pred-parse.txt --gold_out_file gold-parse.txt --gpu 0 --lowercase 1 --replace_num 1
```

## Acknowledgements
Much of our preprocessing and evaluation code is based on the following repositories:  
- [Recurrent Neural Network Grammars](https://github.com/clab/rnng)  
- [Parsing Reading Predict Network](https://github.com/yikangshen/PRPN)  
- [Ordered Neurons](https://github.com/yikangshen/Ordered-Neurons)  

## License
MIT