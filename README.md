# Compound Probabilistic Context-Free Grammars

## Dependencies
The code was tested in `python 3.6` and `pytorch 1.0`. We also require the `nltk` package.

## Data  
The datasets can be downloaded [here](https://drive.google.com/file/d/1bG4_ctFcU63j_tUTMrz7udg2tTBft-bh/view?usp=sharing). This contains the train/validation/test sets, as well as the vocabulary used (`ptb.dict`). If you want to create this from scratch, you can run
```
python process_ptb.py --ptb_path PATH-TO-PTB/parsed/mrg/wsj --output_path DATA-FOLDER
```
where `PATH-TO-PTB` is the location of your PTB corpus and `OUTPUT-FOLDER` is where the processed trees are saved. This will create `ptb-train.txt`, `ptb-valid.txt`, `ptb-test.txt` in `DATA-FOLDER`.

Now run the preprocessing script
```
python preprocess.py --trainfile data/ptb-train.txt --valfile data/ptb-valid.txt --testfile data/ptb-test.txt --outputfile data/ptb
```
See `preprocess.py` for more options (e.g. batch size, vocab size).
Running this will save the following files in the `data/` folder: `ptb-train.pkl`, `ptb-val.pkl`, `ptb-test.pkl`, `ptb.dict`. Here `ptb.dict` is the word-idx mapping, and you can change the output folder/name by changing the argument to `--outputfile`.

## Training
To train the compound PCFG, run
```
python train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path compound-pcfg.pt --gpu 0
```
where `--save_path` is where you want to save the model, and `--gpu 0` is for using the first GPU in the cluster (the mapping from PyTorch GPU index to your cluster's GPU index may vary).

To train the neural PCFG that does not use continuous latent variables, run 
```
python train.py --z_dim 0 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path neural-pcfg.pt --gpu 0
```
Training will take 2-4 days depending on your setup.

## Evaluation
To evaluate the trained model on the test set, run
```
python eval.py --model_file compound-pcfg.pt --data_file data/ptb-test.txt --out_file pred-parse.txt --gpu 0
```
where `--out_file` is where you want to output the predicted parses. This will calculate the F1 scores between the predicted trees and the gold trees in `ptb-test.txt`.

To parse a new set of sentences, run
```
python parse.py --model_file compound-pcfg.pt --data_file sents-to-be-parsed.txt --out_file pred-parse.txt --gpu 0
```
Note that `sents-to-be-parsed.txt` should have one sentence per line, and be preprocessed in a way that roughly matches the processing in `process_ptb.py` (e.g. no punctuation).

## Parsed Datasets
We also provide parsed test sets from the compound/neural PCFGs for further analysis. These can be found under the `data/parsed-data` folder when you download the processed datasets from above:  
- `ptb-test-gold-filtered.txt`: Test set with gold trees with length <= 2 sentences filtered out (this does not affect F1 score since we ignore sentence-level spans for evaluation per convention).  
- `ptb-test-X-runY.txt`: Test set parsed with model `X` that ranked `Y`-th among four different seeds. For example `ptb-test-compound-pcfg-run3.txt` is the test set parsed with the compound PCFG that performed the third best out of four runs.  

## Acknowledgements
Much of our preprocessing and evaluation code is based on the following repositories:  
- [Recurrent Neural Network Grammars](https://github.com/clab/rnng)  
- [Parsing Reading Predict Network](https://github.com/yikangshen/PRPN)  
- [Ordered Neurons](https://github.com/yikangshen/Ordered-Neurons)  

## License
MIT