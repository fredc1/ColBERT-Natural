import base64
import gzip
import json
import os
import sys
import numpy as np
import torch
from colbert.modeling.inference import ModelInference

class LongAnswerCandidate(object):
    """Representation of long answer candidate."""
    
    def __init__(self, contents, index, is_answer, contains_answer):
        self.contents = contents
        self.index = index
        self.is_answer = is_answer
        self.contains_answer = contains_answer
        if is_answer:
            self.style = 'is_answer'
        elif contains_answer:
            self.style = 'contains_answer'
        else:
            self.style = 'not_answer'
        
class Example(object):
    """Example representation. MAY ONLY CALL IF THE EXAMPLE HAS A LONG ANSWER AND A SHORT ANSWER"""
    
    def __init__(self, json_example):
        self.json_example = json_example
        
        # Whole example info.
        self.document_html = self.json_example['document_html'].encode('utf-8')
        self.document_tokens = self.json_example['document_tokens']
        self.question_text = json_example['question_text']

        if len(json_example['annotations']) != 5:
            raise ValueError('Dev set json_examples should have five annotations.')
            
        
        self.long_answers = [a['long_answer'] for a in json_example['annotations'] if a['long_answer']['start_byte'] >= 0 ]
        # self.short_answers = [a['short_answers'][0] for a in json_example['annotations'] if a['long_answer']['start_byte'] >= 0 ]

        
        long_answer_bounds = [(la['start_byte'], la['end_byte']) for la in self.long_answers]
        long_answer_counts = [long_answer_bounds.count(la) for la in long_answer_bounds]
        long_answer = self.long_answers[np.argmax(long_answer_counts)]
        long_answer_text = ' '.join([t['token'] for t in self.json_example['document_tokens'][long_answer['start_token']:long_answer['end_token']]])
        
        # short_answer = self.short_answers[np.argmax(long_answer_counts)]
        # self.short_answer_text = ' '.join([t['token'] for t in self.json_example['document_tokens'][short_answer['start_token']:short_answer['end_token']]])

        self.candidates = self.get_candidates(self.json_example['long_answer_candidates'])
        self.passages = ""
        count = 0
        for candidate in self.candidates:
            if count > 2:
                break

            if candidate.is_answer:
                continue

            self.passages += candidate.contents + "\n"

        self.passages += long_answer_text + "\n"
        

    def get_candidates(self, json_candidates):
        """Returns a list of `LongAnswerCandidate` objects for top level candidates.
    
        Args:
        json_candidates: List of Json records representing candidates.
    
        Returns:
        List of `LongAnswerCandidate` objects.
        """
        candidates = []
        
        for i, candidate in enumerate(json_candidates):

            if candidate['top_level']:
                
                tokenized_contents = ' '.join([t['token'] for t in self.json_example['document_tokens'][candidate['start_token']:candidate['end_token']]])
        
                start = candidate['start_byte']
                end = candidate['end_byte']
                is_answer = np.any([(start == ans['start_byte']) and (end == ans['end_byte']) for ans in self.long_answers])
        
                candidates.append(LongAnswerCandidate(tokenized_contents, len(candidates), is_answer, False))
    
        return candidates
                

        

    
def process_examples(fileobj):
    total = 0
    for l in fileobj:
        if total == 10:
            break
            
        json_example = json.loads(l)
        if not _has_long_answer(json_example):
            continue
        
        example = Example(json_example)
            
        print(f"Here is a search query: {example.question_text}")
        print()
        print()
        print(f"Here is some context from wikipedia that may or may not be relevant\n{example.passages}")
        print()
        print()
        print("Can you use the context to answer the question? If so please answer the question directly in as few words as possible. Otherwise answer: Not enough information.")
        print()
        print()
        total += 1



def _has_long_answer(json_example):
    return sum([ annotation['long_answer']['start_byte'] >= 0 for annotation in json_example['annotations']]) >= 2

# def _has_short_answer(json_example):
        
#     return sum([ annotation['short_answers'][0]['start_byte'] >= 0 for annotation in json_example['annotations'] if annotation['short_answers']]) >= 2

def main(nq_jsonl):
    
    with open(nq_jsonl) as fileobj:
        process_examples(fileobj)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Turn natural questions in jsonl into txt query \t positive \t negative')

    parser.add_argument('--nq_jsonl', type=str, default=None, help='absolute path to your jsonl file')
    args = parser.parse_args()
    
    main(args.nq_jsonl)
