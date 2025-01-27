# Adapted from Google's Natural Question Github Repo
# Curse you google for using 2-space tab

import base64
import gzip
import json
import os
import sys
from tqdm import tqdm
import numpy as np


class Example(object):
    """Example representation."""
    
    def __init__(self, json_example):
        self.json_example = json_example
        
        # Whole example info.
        self.url = json_example['document_url']
        self.document_tokens = self.json_example['document_text'].split()
        self.question_text = json_example['question_text']
        #self.candidates = self.get_candidates(self.json_example['long_answer_candidates'])
        #self.candidates_with_answer = [i for i, c in enumerate(self.candidates) if c.contains_answer]
        
        
        if len(json_example['annotations']) != 1:
            raise ValueError('Train set json_examples should have a single annotation.')
            
        annotation = json_example['annotations'][0]
        self.has_long_answer = annotation['long_answer']['start_token'] >= 0
        self.max_token_len = 0
        
        if not self.has_long_answer:
            self.long_answer_text = ''
        else: 
            long_answer = annotation['long_answer']
            self.long_answer_text = ' '.join(self.document_tokens[int(long_answer['start_token']):int(long_answer['end_token'])])
            self.max_token_len = int(long_answer['end_token']) - int(long_answer['start_token'])

        top_level_candidates = [c for c in self.json_example['long_answer_candidates'] if c['top_level']]

        if len(top_level_candidates) == 0:
            self.alternate_answer_text = "This catches the edgecase of no candidates that I am too lazy to check for."  
        else:  
            candidate = top_level_candidates[0]
            self.alternate_answer_text = ' '.join(self.document_tokens[candidate['start_token']:candidate['end_token']])
            
            self.max_token_len = max(self.max_token_len, int(candidate['end_token']) - int(candidate['start_token']))

            
        
        


def has_long_answer(json_example):
    for annotation in json_example['annotations']:
        if annotation['long_answer']['start_token'] >= 0:
            return True
    return False


def process_examples(fileobj):
    """Reads jsonlines containing NQ examples.
    
    Args:
    fileobj: File object containing NQ examples.
    
    Returns:
    Dictionary mapping example id to `Example` object.
    """
    
    def _load_and_save(f):
        """Read serialized json from `f`, create examples, and add to `examples`."""
        with open(g_output_path, 'w') as tsv_file:
            last_example_w_long_answer = None
            for l in tqdm(f, desc="Extracting"):
                json_example = json.loads(l)
                if not has_long_answer(json_example):
                    continue
                
                example = Example(json_example)

                if last_example_w_long_answer != None:
                    tsv_file.write(f"{example.question_text}\t{example.long_answer_text}\t{last_example_w_long_answer.alternate_answer_text}\n")

                last_example_w_long_answer = example

    
    if g_is_zipped:
        _load_and_save(gzip.GzipFile(fileobj=fileobj))
    else:
        _load_and_save(fileobj)
    

def main(nq_jsonl, tsv_file, gzipped):
    global g_is_zipped
    g_is_zipped = gzipped
    global g_output_path
    g_output_path = tsv_file
    
    with open(nq_jsonl) as fileobj:
        process_examples(fileobj)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Turn natural questions in jsonl into txt query \t positive \t negative')

    parser.add_argument('--nq_jsonl', type=str, default=None, help='absolute path to your jsonl file')
    parser.add_argument('--tsv_file', type=str, default=None, help='absolute path to your tsv output')
    parser.add_argument('--gzipped', action='store_true', help='if you are using gzipped jsonl files as input')
    args = parser.parse_args()
    
    main(args.nq_jsonl, args.tsv_file, args.gzipped)