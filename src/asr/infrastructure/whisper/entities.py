from typing import List, Optional
from loguru import logger

from asr.domain.entities import ASRResponse


class HypothesisBuffer:

    def __init__(self):
        self.commited: List[ASRResponse] = []  # finalized words
        self.buffer: List[ASRResponse] = []     # previous insert
        self.new: List[ASRResponse] = []        # latest insert

        self.last_commited_time = 0
        self.last_commited_word = None
        self.eps = 0.1

    def insert(self, new:List[ASRResponse], offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        shifted = [
                ASRResponse(start=entry.start+offset,
                           end=entry.end+offset,
                           word=entry.word) 
                for entry in new
                ]
        
        filtered = [entry for entry in shifted if entry.start > self.last_commited_time-self.eps]

        if filtered and self.commited:
            if abs(filtered[0].start - self.last_commited_time) < 1:
                # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                max_ngram = min((len(self.commited),len(filtered),5)) + 1

                for n in range(1,max_ngram):  # try largest n-gram first
                    committed_tail = " ".join([self.commited[-i].word for i in range(1,n+1)][::-1])
                    new_head = " ".join(filtered[i-1].word for i in range(1,n+1))
                    if committed_tail == new_head:
                        removed_words = [filtered.pop(0) for _ in range(n)]
                        logger.debug(f"Removed duplicate words: {removed_words}")
                        break
        self.new = filtered

    def flush(self)->Optional[List[ASRResponse]]:
        commit = []
        for new_entry,old_entry in zip(self.new,self.buffer):
            if new_entry.word.lower() == old_entry.word.lower():
                commit.append(new_entry)
            else:
                break
        
        # update internal states
        self.last_commited_time = commit[-1].end if commit else self.last_commited_time
        self.last_commited_word = commit[-1].word if commit else self.last_commited_word

        len_commited = len(commit)
        self.buffer = self.new[len_commited:]
        self.new = []
        self.commited.extend(commit)
        return commit

    def pop_commited(self, time:float):
        while self.commited and self.commited[0].end <= time:
            self.commited.pop(0)

    def complete(self):
        return self.buffer.copy()