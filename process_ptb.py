import os
import re
import sys
import argparse
import nltk
from nltk.corpus import ptb
import os
from pathlib import Path


def get_data_ptb(root, output):
  # tag filter is from https://github.com/yikangshen/PRPN/blob/master/data_ptb.py
  word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 
               'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 
               'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'WDT', 'WP', 'WP$', 'WRB']
  currency_tags_words = ['#', '$', 'C$', 'A$']
  ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
  punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
  punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', 
                       '-', '?', '!', '...', '-LCB-', '-RCB-']
  train_file_ids = []
  val_file_ids = []
  test_file_ids = []
  train_section = ['02', '03', '04', '05', '06', '07', '08', '09', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
  val_section = ['22']
  test_section = ['23']

  for dir_name, _, file_list in os.walk(root, topdown=False):
    if dir_name.split("/")[-1] in train_section:
      file_ids = train_file_ids
    elif dir_name.split("/")[-1] in val_section:
      file_ids = val_file_ids
    elif dir_name.split("/")[-1] in test_section:
      file_ids = test_file_ids
    else:
      continue
    for fname in file_list:
      file_ids.append(os.path.join(dir_name, fname))
      assert(file_ids[-1].split(".")[-1] == "mrg")
  print(len(train_file_ids), len(val_file_ids), len(test_file_ids))

  def del_tags(tree, word_tags):    
    for sub in tree.subtrees():
      for n, child in enumerate(sub):
        if isinstance(child, str):
          continue
        if all(leaf_tag not in word_tags for leaf, leaf_tag in child.pos()):
          del sub[n]

  def save_file(file_ids, out_file):
    sens = []
    trees = []
    tags = []
    f_out = open(out_file, 'w')
    for f in file_ids:
      sentences = ptb.parsed_sents(f)
      for sen_tree in sentences:
        orig = sen_tree.pformat(margin=sys.maxsize).strip()
        c = 0
        while not all([tag in word_tags for _, tag in sen_tree.pos()]):
          del_tags(sen_tree, word_tags)
          c += 1
          if c > 10:
            assert False
        out = sen_tree.pformat(margin=sys.maxsize).strip()          
        while re.search('\(([A-Z0-9]{1,})((-|=)[A-Z0-9]*)*\s{1,}\)', out) is not None:
          out = re.sub('\(([A-Z0-9]{1,})((-|=)[A-Z0-9]*)*\s{1,}\)', '', out)
        out = out.replace(' )', ')')
        out = re.sub('\s{2,}', ' ', out)
        f_out.write(out + '\n')
    f_out.close()

  save_file(train_file_ids, os.path.join(output, "ptb-train.txt"))
  save_file(val_file_ids, os.path.join(output, "ptb-valid.txt"))
  save_file(test_file_ids, os.path.join(output, "ptb-test.txt"))

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ptb_path', help='Path to parsed/mrg/wsj folder', type=str, 
                        default='PATH-TO-PTB/parsed/mrg/wsj')
    parser.add_argument('--output_path', help='Path to save processed files',
                        type=str, default='data')
    args = parser.parse_args(arguments)
    get_data_ptb(args.ptb_path, args.output_path)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


