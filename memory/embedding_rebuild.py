"""Rebuild stored vectors after an embedding-space change."""
from __future__ import annotations

import argparse
import json

from memory.embedding_pipeline import current_embedding_metadata, get_embedding_pipeline
from memory.memory_repository import ChromaMemoryRepository


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--path',default='./memory_db'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); p.add_argument('--workers',type=int,default=1); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--progress',action='store_true'); a=p.parse_args(argv)
    import os; os.environ['PRIMA_EMBEDDING_MISMATCH']='readonly' if a.dry_run else 'rebuild'; repo=ChromaMemoryRepository(a.path)
    notes=repo.list(); meta=current_embedding_metadata(); report={'status':'dry-run' if a.dry_run else 'rebuilt','count':len(notes),'backend':meta,'workers':a.workers,'batch_size':a.batch_size}
    if not a.dry_run:
        pipe=get_embedding_pipeline()
        for i,note in enumerate(notes,1):
            old=note.retrieval_metadata.get('embedding_text')
            embedded=pipe.embed_memory(note.content, embedding_text=old)
            repo.update(note.with_updates(embedding=embedded.vector,retrieval_metadata={**note.retrieval_metadata,**embedded.metadata}))
            if a.progress: print(f'{i}/{len(notes)}')
    print(json.dumps(report,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
