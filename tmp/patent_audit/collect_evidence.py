import ast
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

ROOT = Path('C:/PRIMA_next')
TMP = ROOT / 'tmp/patent_audit'
original = Path('C:/Users/Yuv Nahar/Downloads/Indian_patent_guidelines_CRI-AIML.pdf')
guide = ROOT / 'output/pdf/Indian_CRI_AI_ML_Plain_Language_Guide.pdf'
highlights = []
reader = PdfReader(guide)
with pdfplumber.open(guide) as doc:
    for n, page in enumerate(reader.pages):
        for ref in page.get('/Annots', []):
            a = ref.get_object()
            if a.get('/Subtype') != '/Highlight':
                continue
            q = [float(x) for x in a.get('/QuadPoints', [])]
            parts = []
            for start in range(0, len(q), 8):
                xs, ys = q[start:start+8:2], q[start+1:start+8:2]
                if len(xs) != 4:
                    continue
                box = (max(0,min(xs)-0.5), max(0,float(page.mediabox.height)-max(ys)-0.5), min(float(page.mediabox.width),max(xs)+0.5), min(float(page.mediabox.height),float(page.mediabox.height)-min(ys)+0.5))
                parts.append(doc.pages[n].crop(box).extract_text() or '')
            highlights.append({'id':f'H{len(highlights)+1:02}', 'guide_page':n+1, 'text':' '.join(parts), 'quadpoints':q})
assert len(highlights) == 56, len(highlights)
(TMP/'highlights.json').write_text(json.dumps(highlights,indent=2),encoding='utf-8')
sources = []
for p in [original,guide]:
    sources.append({'path':str(p), 'bytes':p.stat().st_size,'pages':len(PdfReader(p).pages),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(TMP/'source_provenance.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
(TMP/'original_pages.json').write_text(json.dumps([{'page':n+1,'text':p.extract_text()} for n,p in enumerate(PdfReader(original).pages)],indent=2),encoding='utf-8')

# Execute the actual compressor class body in isolation, avoiding model imports.
tree = ast.parse((ROOT/'memory/retrieval/context_compressor.py').read_text())
selected = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    text: str
    provenance: dict
ns = {'dataclass':dataclass,'replace':replace,'Any':Any,'EvidenceItem':Evidence}
exec(compile(ast.Module(body=selected,type_ignores=[]),'context_compressor.py','exec'),ns)
c = ns['ContextCompressor']()
e = Evidence('E1','S1','Valve is open only when sensor is not faulty',{})
r = c.compress((e,),3)
assert r.evidence[0].text == 'Valve is open'
assert r.citation_map['E1'] == 'S1'
assert r.compressed_tokens == 3 and r.original_tokens == 9
try:
    c.compress((e,),0)
    raise AssertionError('missing boundary check')
except ValueError:
    pass
assert ns['ContextCompressor'](enabled=False).compress((e,),3).evidence[0].text == e.text
probe = {'scope':'Actual compressor class/function AST executed with a minimal frozen evidence fixture; not an integration test or quality benchmark.', 'checks_passed':4,'input':e.text,'budget':3,'retained':r.evidence[0].text,'implication':'Source-ID preservation does not preserve conditions or negation. Counts are whitespace words.', 'focused_suite':'Attempted five-file pytest suite returned no result and was interrupted; no fresh integration-test pass claimed.'}
(TMP/'mechanism_probe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
print(json.dumps({'highlights':len(highlights),'sources':sources,'probe':probe},indent=2))
