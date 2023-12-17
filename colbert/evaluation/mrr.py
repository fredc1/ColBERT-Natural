import base64
import gzip
import json
import os
import sys
import numpy as np
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
    """Example representation."""
    
    def __init__(self, json_example):
        self.json_example = json_example
        
        # Whole example info.
        self.document_html = self.json_example['document_html'].encode('utf-8')
        self.document_tokens = self.json_example['document_tokens']
        self.question_text = json_example['question_text']

        if len(json_example['annotations']) != 5:
            raise ValueError('Dev set json_examples should have five annotations.')
        self.has_long_answer = sum([ annotation['long_answer']['start_byte'] >= 0 for annotation in json_example['annotations']]) >= 2
        
        
        self.long_answers = [a['long_answer'] for a in json_example['annotations'] if a['long_answer']['start_byte'] >= 0 and self.has_long_answer ]
        
        if self.has_long_answer:
            long_answer_bounds = [(la['start_byte'], la['end_byte']) for la in self.long_answers]
            long_answer_counts = [long_answer_bounds.count(la) for la in long_answer_bounds]
            long_answer = self.long_answers[np.argmax(long_answer_counts)]
            self.long_answer_candidates_idx = int(long_answer['candidate_index'])
        
        else:
            self.long_answer_candidates_idx = -1

        self.candidates = self.get_candidates(
        self.json_example['long_answer_candidates'])
        
        self.candidates_with_answer = [i for i, c in enumerate(self.candidates) if c.contains_answer]

    def get_candidates(self, json_candidates):
        """Returns a list of `LongAnswerCandidate` objects for top level candidates.
    
        Args:
        json_candidates: List of Json records representing candidates.
    
        Returns:
        List of `LongAnswerCandidate` objects.
        """
        candidates = []
        top_level_candidates = [c for c in json_candidates if c['top_level']]
        for candidate in top_level_candidates:
            tokenized_contents = ' '.join([t['token'] for t in self.json_example['document_tokens'][candidate['start_token']:candidate['end_token']]])
    
            start = candidate['start_byte']
            end = candidate['end_byte']
            is_answer = self.has_long_answer and np.any([(start == ans['start_byte']) and (end == ans['end_byte']) for ans in self.long_answers])
            contains_answer = self.has_long_answer and np.any([(start <= ans['start_byte']) and (end >= ans['end_byte']) for ans in self.long_answers])
    
            candidates.append(LongAnswerCandidate(tokenized_contents, len(candidates), is_answer, contains_answer))
    
        return candidates

    def infer_ranking(self, model, bsize, amp):
        inference = ModelInference(model, amp=amp)

        with torch.no_grad():
            passages = [lac.contents for lac in self.candidates]
            Q = inference.queryFromText([self.question_text])
            D_ = inference.docFromText(passages, bsize=bsize)
            scores = model.score(Q, D_).cpu()
            scores = scores.sort(descending=True)
            ranked = scores.indices.tolist()
            self.rr = 0
            for i, idx in enumerate(ranked):
                if i == 10:
                    break
                if idx == self.long_answer_candidates_idx:
                    self.rr = 1.0/(i+1)
                

        
class NQ_Validator():
    """ A class to test colbert model on non-training data """

    def __init__(self, model, val_json_file, amp, bsize):
        self.bsize = bsize
        self.amp = amp
        self.model = model
        self.mrr = 0
        with open(val_json_file) as fileobj:
            self.process_examples(fileobj)
        
    def process_examples(self, fileobj):
        total_rr = 0.0
        total = 0
        
        for l in fileobj:
            if total == 1000:
                break
                
            json_example = json.loads(l)
            if not _has_long_answer(json_example):
                continue
            
            example = Example(json_example)
            example.infer_ranking(model, self.bsize, self.amp)

            total_rr += example.rr
            total += 1

        self.mrr = total_rr/total

    def get_mrr(self):
        return self.mrr
    
    @staticmethod
    def _has_long_answer(json_example):
        return sum([ annotation['long_answer']['start_byte'] >= 0 for annotation in json_example['annotations']]) >= 2
    
