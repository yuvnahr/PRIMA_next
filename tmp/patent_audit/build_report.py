import csv
import hashlib
import html
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from research_content import BIBLIOGRAPHY, INTRO, OPPORTUNITIES, PATENTS, PROTOCOL

ROOT = Path('C:/PRIMA_next')
TMP = ROOT/'tmp/patent_audit'
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
BASE = 'PRIMA_Patent_Research_Foundation'
inv = json.loads((TMP/'source_inventory.json').read_text(encoding='utf-8'))
highlights = json.loads((TMP/'highlights.json').read_text(encoding='utf-8'))
byid = {o['id']:o for o in OPPORTUNITIES}
assert list(byid) == [f'R{i:02}' for i in range(1,25)]

def validate_anchor(s):
    p, n = s.rsplit(':',1)
    f = ROOT/p
    assert f.is_file(),s
    if f.suffix != '.ipynb':
        assert 1 <= int(n) <= len(f.read_text(encoding='utf-8-sig').splitlines()),s
for o in OPPORTUNITIES:
    for a in o['code']:
        validate_anchor(a)
for f in inv['files']:
    assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest() == f['sha256'], f['path']
assert len(inv['files']) == 385 and sum(x['lines'] for x in inv['files']) == 44720

def classify(p):
    s=p.lower()
    name=Path(s).name
    if name=='__init__.py': return 'R24','Export/package support; contribution depends on implemented modules.'
    if s.startswith('tests/'):
        targets=[('identity','R10'),('confidence','R12'),('temporal','R11'),('replica','R07'),('campaign','R07'),('kaggle','R07'),('goemotions','R19'),('correction','R04'),('reflection','R04'),('retrieval','R02'),('graph','R17'),('evolution','R18'),('importance','R08'),('admission','R08'),('novelty','R08'),('recurrence','R08'),('maintenance','R01'),('embedding','R05'),('action','R14'),('tool','R14'),('state','R06'),('affect','R15')]
        return next(((r,'Verification support for the named research family; no standalone invention inferred.') for k,r in targets if k in s),('R22','Test/fixture support for reproducibility and runtime behavior.'))
    if s.startswith('memory/'):
        if 'context_compressor' in s: r='R03'
        elif 'identity' in s or 'memory_context' in s: r='R10'
        elif 'semantic_representation' in s: r='R18'
        elif '/event_memory/' in s: r='R16'
        elif '/evolution/' in s: r='R18'
        elif any(k in s for k in ['embedding','memory_repository','repository_factory']): r='R05'
        elif '/graph/' in s: r='R01' if any(k in s for k in ['builder','repository']) else 'R17'
        elif '/retrieval/' in s: r='R11' if 'temporal' in s else 'R12' if 'confidence' in s else 'R02'
        elif any(k in s for k in ['importance','salience_manager']): r='R08'
        elif any(k in s for k in ['consolidation','retention','forgetting']): r='R09'
        elif '/maintenance/' in s or 'memory_index' in s: r='R01'
        elif 'lineage' in s: r='R18'
        else: r='R01'
    elif s.startswith('benchmarks/campaign/'): r='R22' if any(k in s for k in ['comparison','aggregation','manifest']) else 'R07'
    elif s.startswith('benchmarks/goemotions/'): r='R19'
    elif s.startswith('benchmarks/'): r='R16' if 'semantic_representation' in s else 'R02' if 'retrieval_validation' in s else 'R22'
    elif s.startswith('evaluation/'): r='R12' if 'calibration' in s else 'R04' if 'reflection' in s else 'R02' if 'retrieval' in s else 'R22'
    elif s.startswith('affect/'): r='R19' if any(k in s for k in ['classifiers/','classifier_factory','emotion_classifier','taxonomy','taxonomies/']) else 'R15'
    elif s.startswith('state/'): r='R06'
    elif s.startswith(('action/','tools/')): r='R13' if any(k in s for k in ['execution_result','tool_invocation']) else 'R14'
    elif s.startswith(('planning/','world/')): r='R20'
    elif s.startswith('uncertainty/'): r='R12'
    elif s.startswith('reflection/'): r='R13' if 'rule_extractor' in s else 'R04'
    elif s.startswith('reasoning/'): r='R03' if 'evidence_integrator' in s else 'R04' if 'reflection' in s else 'R02'
    elif s.startswith('workflow/'): r='R04' if any(k in s for k in ['correction','orchestration']) else 'R03' if 'answer_generation' in s else 'R21'
    elif s.startswith('runtime/'): r='R21'
    elif s.startswith('security/'): r='R21'
    elif s.startswith('llm/'): r='R21' if any(k in s for k in ['prompt_builder','generation_config']) else 'R23'
    elif s.startswith(('events/','monitoring/','config/')) or s=='main.py': r='R23'
    else: r='R24'
    return r,'Implementation or schema in this family; consult symbols and charter, not an assertion of standalone patentability.'

deep = set(a.rsplit(':',1)[0] for o in OPPORTUNITIES for a in o['code']) | {
'memory/retrieval/hybrid_fusion.py','memory/retrieval/reranker.py','memory/retrieval/query_analysis.py','memory/retrieval/sparse_strategy.py','memory/retrieval/dense_strategy.py',
'memory/evolution/semantic_abstraction_engine.py','memory/event_memory/event_segmenter.py','affect/affect_evolution.py','affect/emotion_profile.py','affect/pad_model.py',
'tools/tool_validator.py','security/redaction.py','security/output_validator.py','planning/action_selector.py','planning/goal_selector.py','reasoning/controller.py',
'reasoning/sufficiency_verifier.py','reasoning/stopping_policy.py','reasoning/information_need_generator.py','reflection/reflection_confidence.py','reflection/rule_extractor.py',
'llm/provider.py','llm/response_parser.py','benchmarks/goemotions/README.md','benchmarks/locomo/README.md','benchmarks/hotpotqa/README.md',
'README.md','pyproject.toml','setup.ps1','.gitmodules','config/retrieval.yaml','config/memory_importance.yaml','docs/benchmarks/RESULT_CLAIM_POLICY.md',
'evaluation/results/final/embedding_ablation_results.json','evaluation/results/final/representation_ablation_results.json'}
rows=[]
for f in inv['files']:
    rid,role=classify(f['path'])
    defs=f['definitions']
    symbols='; '.join(f"{d['name']}:{d['line']}" for d in defs if d['kind']=='ClassDef' or '.' not in d['name'])[:1600]
    rows.append(dict(path=f['path'],asset_type='owned_python',primary_family=rid,research_rank=int(rid[1:]),tier=byid[rid]['tier'],
        disposition=role,review_depth='Relevant implementation/call-path read plus AST inventory' if f['path'] in deep else 'Structural AST inventory; family-level review',
        lines=f['lines'],bytes=(ROOT/f['path']).stat().st_size,sha256=f['sha256'],module_role=f['doc'],symbols=symbols,
        related_families='R01;R05;R06;R08;R09;R16;R18' if f['path'].startswith('memory/') else 'See charter dependencies and module matrix',
        standalone_patent_status='Not established; mapping is a research disposition'))
paths=subprocess.run(['rg','--files','--hidden','-g','!.git/**','-g','!.venv/**','-g','!**/external/**','-g','!output/**','-g','!tmp/**','-g','!logs/**','-g','!**/__pycache__/**','-g','!.pytest_cache/**'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
paths += ['evaluation/results/final/embedding_ablation_results.json','evaluation/results/final/representation_ablation_results.json']
owned_py={f['path'] for f in inv['files']}
for name in sorted({s.replace('\\','/') for s in paths}):
    p=ROOT/name
    if name in owned_py or p.suffix=='.py' or not p.is_file(): continue
    if p.name.startswith('.env'): continue
    if any(k in name.lower() for k in ['secret','credential','token_cache']): continue
    if p.suffix=='.ipynb': rid,role='R07','Notebook campaign infrastructure; source cells inspected, not executed.'
    elif name.startswith('evaluation/results/'): rid,role='R22','Saved evaluation output; historical evidence only, not a fresh run.'
    elif name.startswith(('benchmarks/campaign/','docs/kaggle')): rid,role='R07','Campaign configuration/documentation; claims require measured reproduction.'
    elif name.startswith(('tests/fixtures/','benchmarks/')): rid,role='R24','Dataset/fixture/configuration or benchmark documentation; declare provenance and split.'
    elif name.startswith(('config/','.github/')): rid,role='R23','Configuration/build/CI support; not a standalone mechanism.'
    elif name.startswith('docs/'): rid,role='R24','Project documentation/presentation; secondary to current source.'
    else: rid,role='R24','Project metadata/setup/license/data support; not assumed team-original invention.'
    b=p.stat().st_size
    rows.append(dict(path=name,asset_type='non_python_asset',primary_family=rid,research_rank=int(rid[1:]),tier=byid[rid]['tier'],disposition=role,
        review_depth='Relevant content/structure inspected' if name in deep or p.suffix=='.ipynb' else 'Path/type inventory; contents not exhaustively reviewed',lines='',bytes=b,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest() if b<2_000_000 else 'Not hashed: large supporting asset',module_role='',symbols='',related_families='See module matrix',standalone_patent_status='No standalone conclusion'))
for path in ['benchmarks/locomo/external/','benchmarks/locomo-v2/external/','benchmarks/hotpotqa/external/']:
    rows.append(dict(path=path,asset_type='third_party_tree_boundary',primary_family='R24',research_rank=24,tier='D',disposition='Third-party subtree; excluded from owned-source count and invention attribution.',review_depth='Provenance/README boundary review; no exhaustive external-code audit',lines='',bytes='',sha256='',module_role='External benchmark/data implementation',symbols='',related_families='R07;R22',standalone_patent_status='External work; not attributed to team'))
assert len({r['path'] for r in rows})==len(rows)
assert {r['path'] for r in rows if r['asset_type']=='owned_python'}==owned_py
inventory_path=OUT/(BASE+'_File_Ledger.csv')
with inventory_path.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rank_path=OUT/(BASE+'_Ranked_Opportunities.csv')
rankrows=[]
for o in OPPORTUNITIES:
    rankrows.append({'rank':int(o['id'][1:]),'id':o['id'],'title':o['title'],'tier':o['tier'],'technical_effect_fit':o['technical'],'evidence_readiness':o['readiness'],'prior_art_crowding':o['crowding'],'effort':o['effort'],'current_behavior':o['current'],'gap':o['gap'],'proposed_mechanism':o['hypothesis'],'example':o['example'],'first_experiment':o['first'],'experiment_and_baselines':o['experiment'],'metrics_and_ablations':o['metrics'],'proposed_decision_gate':o['gate'],'code_anchors':'; '.join(o['code']),'guidelines_original_pages':o['pages'],'prior_art_ids':o['art'],'disclosure_needed':o['disclosure'],'novelty_nonobviousness':'Unproven; targeted full search and professional assessment required','owner_to_assign':'','team_decision':'','experiment_run_ids':''})
with rank_path.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rankrows[0]));w.writeheader();w.writerows(rankrows)

module_notes={
'runtime':('R21; R01; R06','Canonical lifecycle, dependency injection, ten valid routes and maintenance ownership. Strong as the integration surface for a narrow mechanism; generic orchestration alone is weak.'),
'workflow':('R04; R03; R06; R13','Evidence acquisition, correction reinsertion, generation/validation and commits. Trace this caller before attributing an effect to a dormant helper.'),
'memory':('R01-R03; R05; R08-R11; R16-R18','Largest research surface: repository/index, retrieval strategies, representation, event/identity processing, maintenance and evolution. The per-file ledger resolves every submodule.'),
'affect':('R15; R19','Existing encoder/legacy classifier, PAD/momentum/history, salience and hints. Technical control effect is not demonstrated by the canonical retrieval path.'),
'reasoning':('R02-R04; R17','Bounded deterministic evidence acquisition, integration and sufficiency. Lexical coverage is not entailment; isolate cost and unsupported-answer behavior.'),
'planning':('R20; R04','Deterministic goals, action selection, plan evaluation and replanning. Supports a verified execution experiment rather than a broad cognition claim.'),
'reflection':('R04; R13','Triggers, acceptance, adapters, templates, lineage and rule extraction. Separate actual outcome improvement from heuristic confidence.'),
'action':('R13; R14','Sequential execution, policy and audit records. Success status is distinct from a verified external postcondition.'),
'tools':('R14; R13','Routing, registry, schema validation and execution. Real isolation/cancellation/effect semantics must be established before stronger safety claims.'),
'world':('R20','Symbolic predictions and simulation context. No learned physical-world predictor established.'),
'uncertainty':('R12; R04','Static aggregation/calibration profiles and execution gate. Fit/validate real outcome probabilities before using confidence as evidence.'),
'state':('R06; R10; R15','Cognitive/emotional state and optimistic process-local ownership. Scope, concurrency and persistence boundaries matter.'),
'events':('R23; R06','In-process events, publisher/subscriber and maintenance event contracts. Useful instrumentation, no durable broker guarantee.'),
'security':('R21; R14','Allowlist, sanitation, redaction and validation support. Regex and labels are not formal isolation.'),
'llm':('R23; R21; R03','Provider transport, schema/generation contracts, parsing, prompt assembly and rate limits. Reuse for controlled experiments.'),
'config':('R23','Runtime mode/environment/settings. Parameters define the baseline; parameter changes alone are weak research.'),
'monitoring':('R23; R22','Health/metrics/tracing. Measure actual work and overhead; instrumentation is enabling infrastructure.'),
'evaluation':('R22; R02; R04; R12','Metrics, synthetic generators and ablation runners. Distinguish analytic proxies from real measurements; preserve gold outside runtime.'),
'benchmarks':('R07; R19; R22','Owned adapters and campaign, result/provenance tooling and emotion training. Separate third-party data and supplied/oracle contexts.'),
'tests':('R22; all families','Regression evidence and fixtures. Structural coverage is complete; a test file is not proof that its tests were run in this audit.'),
'scripts':('R24','Complexity-check entry point. Support only.'),
'main.py':('R23','Entry-point wiring. No independent technical mechanism established.')}
rootstats=defaultdict(lambda:[0,0])
for f in inv['files']:
    t=f['path'].split('/')[0];rootstats[t][0]+=1;rootstats[t][1]+=f['lines']

parts=[INTRO]
parts.append('## 6. Ranked shortlist and full research map\n\nA/B/C/D indicate research priority only. Every row has unproven novelty. The CSV has editable ownership and decision fields.\n')
parts.append('| Rank | Research family | Tier | Technical fit | Readiness | Crowding |\n|---|---|---|---|---|---|')
for o in OPPORTUNITIES:
    parts.append(f"| {o['id']} | {o['title']} | {o['tier']} | {o['technical']} | {o['readiness']} | {o['crowding']} |")
parts.append('\n## 7. Module coverage and implementation implications\n\nThis matrix accounts for every owned Python root. The ledger expands it to every file; interdependent research families overlap, so their counts must not be added as separate inventions.\n')
parts.append('| Module | Files / lines | Research links | Assessment |\n|---|---|---|---|')
for k in module_notes:
    n,l=rootstats[k];ids,note=module_notes[k]
    parts.append(f'| {k} | {n} / {l:,} | {ids} | {note} |')
parts.append('\nNon-Python scope: notebook cells and campaign/configuration source, repository setup/CI/license metadata, project explanations/presentations, saved results and benchmark fixture/data boundaries. Documentation can suggest a question but cannot override current source. Third-party models/datasets and external submodules require provenance, not team invention credit. Logs/caches/local secrets/generated outputs are excluded from exhaustive invention review.\n')
parts.append(PROTOCOL)
parts.append('## 14. Opportunity charters\n\nEach charter identifies a small proposed mechanism, a standard alternative, a falsifiable technical effect and original-guideline pages. R01-R24 are research IDs, not patent numbers. A source-code anchor identifies implementation evidence; proposed features are explicitly labeled.\n')
for o in OPPORTUNITIES:
    parts.append(f"\n<!-- pagebreak -->\n## {o['id']}. {o['title']}\n\nTier {o['tier']} | Technical fit: {o['technical']} | Readiness: {o['readiness']} | Prior-art crowding: {o['crowding']} | Effort: {o['effort']}\n")
    parts.append('**Code anchors:** '+ '; '.join(f'[{a}](<{ROOT.as_posix()}/{a}>)' for a in o['code'])+'.\n')
    for label,key in [('Implemented now','current'),('Problem to investigate','gap'),('Proposed mechanism','hypothesis'),('Plain example','example'),('Experiment and baselines','experiment'),('Metrics and ablations','metrics'),('Proposed go/no-go gate','gate'),('First task','first'),('Disclosure needed','disclosure')]:
        parts.append(f"**{label}.** {o[key]}\n")
    parts.append(f"**Guideline connection:** original pp. {o['pages']}; technical effect, separate statutory tests and enablement remain necessary. **Closest screen references:** {o['art']}. Novelty and non-obviousness remain unproven.\n")

parts.append('<!-- pagebreak -->\n## 15. Primary research screen\n\nDates below are first arXiv submissions where available, or the stated publication month. They are not an assessment of every earlier public version. Abstract-level scope was used for this screen; full-method comparison remains a team task. Descriptions identify crowding, not established anticipation of a final claim.\n')
for rid,title,date,url,note,rel in BIBLIOGRAPHY:
    parts.append(f'### {rid}. [{title}]({url})\n\nFirst listed public version: {date}. Relevant families: {rel}. {note}\n')
parts.append('## 16. Patent-publication screen and preliminary feature comparison\n\nThese are published records used as technical disclosures. Selected claim passages were inspected; no legal-status, validity, enforceability, Indian-family or freedom-to-operate conclusion is made. Publication dates below must not be confused with priority dates. The final mechanism needs a full feature chart and family search.\n')
for rid,pub,date,url,title,features,diff in PATENTS:
    parts.append(f'### {rid}. [{pub}: {title}]({url})\n\nPublication: {date}. **Inspected feature overlap:** {features}\n\n**Research consequence:** {diff}\n')
parts.append('## 17. Your 56 highlights: requirement-to-research crosswalk\n\nGuide p. means the annotated 27-page study guide; source pp. means the original 67-page guidelines. Short topics below identify your highlights; they are not verbatim quotations of the original law. Adjacent quadpoint crops can contain fragments, so the original provision/page controls. All 56 highlight IDs are accounted for.\n')
topics={
1:('Useful app is not enough','9, 16-23','All','Require eligible implemented contribution and every ordinary patent test.'),
2:('Four separate 3(k) exclusions','9-10, 23-39','All','Check each applicable exclusion separately.'),
3:('Abstract math versus implementation','23-34','R01-R14','Describe executable components and technical operation, not a formula alone.'),
4:('How and best method','20-23','All','Write reproducible structures, steps, failures and best implementation.'),
5:('Architecture versus data contribution','53-61','R05, R08, R19','Disclose what actually supplies the effect.'),
6:('Human inventorship','50','All','Record human conception and AI assistance.'),
7:('AI/ML/DL meaning','6, 49-50','R19, R20','Do not call fixed rules a learned neural architecture.'),
8:('Architecture/layers/parameters','53-61','R19','Describe essential model/training details if claimed.'),
9:('Attention/embeddings/time','53-61','R05, R11, R19','Explain actual representation and temporal mechanism.'),
10:('Preprocessing and segmentation','55-56, 59-61','R10, R16, R19','Connect raw input transformations to final effect.'),
11:('Accuracy/recall/errors/robustness','53-61','All experimental tracks','Use appropriate held-out metrics and failure accounting.'),
12:('Novelty steps 1-6','16-17','All','Compare essential features and earlier public disclosures.'),
13:('Novelty documentation step 7','16-17','All','Save exact prior-art passages, dates and reasons.'),
14:('Non-obviousness also required','17-19','All','Technical advance/economic value does not replace inventive step.'),
15:('Five-step inventive assessment','18-19','All','Identify skilled person, knowledge, concept, differences and obviousness.'),
16:('Industrial applicability','20','All','Specify a workable technical deployment.'),
17:('Internal computation','39-41','R01, R02, R07','Measure actual operations and processor time.'),
18:('Storage access','39-41','R01, R05','Measure I/O and access latency beyond ordinary caching.'),
19:('Memory management','39-41','R01, R08, R09','Measure storage/maintenance and preserved quality.'),
20:('Database operation','39-41','R01, R05, R06','Explain indexing, consistency and recovery mechanisms.'),
21:('Compression/expansion','39-41','R03','Use actual tokens/bytes and original-source fidelity.'),
22:('Authentication/privacy/security','39-41','R14, R21','Require enforceable mechanisms; labels are insufficient.'),
23:('Resource allocation','39-41','R01, R02, R07','Measure resource-bound scheduling against standard queues.'),
24:('Error detection/correction','39-41','R04, R06','Measure actual errors/recovery, not score changes.'),
25:('Machine control','39-41','R14, R20','No actual physical control established in this repo.'),
26:('Signal handling','39-41','R19 conditional','No new signal-processing mechanism established.'),
27:('Real-time monitoring/control','39-41','R07, R14 conditional','Measure deadlines and real controls if claimed.'),
28:('Embedded control','39-41','No current direct implementation','Do not add a hardware label without actual integration.'),
29:('Autonomous vehicle/drone','39-41','No current direct implementation','Outside present implementation; not a candidate by analogy alone.'),
30:('Medical image analysis','39-41','No current direct implementation','Emotion classification is not evidence of medical imaging innovation.'),
31:('Distributed/cloud systems','39-41','R06, R07','Distinguish process-local behavior from distributed guarantees.'),
32:('Technical simulation','39-41','R20 conditional','Symbolic predictions need a technical process and validation.'),
33:('Efficient ML/architecture','39-41','R05, R19','Require a specific new mechanism and measured compute/quality.'),
34:('Technical signal processing','39-41','No current direct mechanism','Do not infer a signal invention from existing encoders.'),
35:('Human-reasoning imitation','41-43','R04, R20','Ground control in technical operation and evidence.'),
36:('Mental/game rules','9, 41-43','R20, R21','A thinking/workflow rule is not enough.'),
37:('Ordinary automation','41-43','R21, R23','Show an additional technical mechanism/effect.'),
38:('Standard lookup','41-43','R02, R11, R17','Beat ordinary retrieval with fair cost/quality accounting.'),
39:('Preference personalization','41-43','R08, R15','Empathy/preferences alone are a weak technical basis.'),
40:('Algorithm flowchart','46','All','Enabled technical implementation and effect matter.'),
41:('Program-per-se flowchart','47-48','All','Interpret whole claim and assess essential technical character.'),
42:('AI-assisted versus AI-generated','50','All','Keep the human contribution record.'),
43:('Named parts without integration','50-53','All','Teach concrete connections and operation, not box names.'),
44:('Transformation and validating evidence','53-54','R03, R16, R19','Reproduce the input-output transformation and technical result.'),
45:('Data/model/training/reliability','54-55','R08, R12, R19','Specify essential data and training traits and reliable outcomes.'),
46:('Preprocessing connection','55-56','R05, R10, R16, R19','Disclose every material step and its role.'),
47:('RL state/action/reward','56-57','R02, R04, R08 if learned','No RL claim without its actual observation/action/reward design.'),
48:('Internal hardware/system performance','58','R01-R07','Explain integration and measure actual internal metrics.'),
49:('Essential dataset traits','59','All learning tracks','Record domain, diversity, labels and defining traits.'),
50:('Learning implementation versus data','60-61','R05, R08, R19','Match disclosure to the claimed inventive contribution.'),
51:('Prompt/workflow reading aid','31-43, 49-61','R04, R21','Guide synthesis, not a dedicated prompt patent rule.'),
52:('RAG/memory reading aid','35-43, 58-61','R01-R03, R05-R11','Ordinary lookup differs from an implemented technical improvement.'),
53:('Fine-tuning/data reading aid','53-61','R08, R19','Generic tuning is weak; disclose essential mechanism/data traits.'),
54:('Agents/control reading aid','56-58','R04, R13, R14, R20','Explain actual observation, action and effect interface.'),
55:('Architecture connection reading aid','20-23, 50-53','All','A memory/reasoning/learning block diagram is not enablement.'),
56:('Labels do not change tests','16-23, 34-43, 49-61','All','Apply all statutory and disclosure checks to the final mechanism.')}
assert set(topics)==set(range(1,57))
parts.append('| Highlight / guide p. | Topic | Original source pp. | Research action / family |\n|---|---|---|---|')
for h in highlights:
    n=int(h['id'][1:]);topic,pages,fams,action=topics[n]
    parts.append(f"| {h['id']} / {h['guide_page']} | {topic} | {pages} | {fams}: {action} |")
parts.append('## 18. Decision records and ready-to-use team templates\n\nFor each surviving candidate, complete the following fields in a shared record. Keep failed candidates and why they failed. This prevents a later patent narrative from erasing important negative evidence.\n')
for label,desc in [
('Candidate and owner','Research ID, human owner, dated version, technical collaborators and reviewer.'),
('Problem and mechanism','One observable technical bottleneck/failure; the smallest proposed operations and data structures; what currently exists and what is new.'),
('Closest alternatives','Current code, standard baseline, closest full papers/patent families and exact feature chart.'),
('Predicted effect','Primary resource/correctness metric, practical minimum effect, quality tolerance and scope assumptions.'),
('Experimental identity','Run manifests, source/model/dataset/config hashes, hardware, splits, seed, budgets and cache regime.'),
('Outcome and uncertainty','All-case results, paired clustered intervals, failures, ablations, cost and limits.'),
('Enablement','Best implementation, pseudocode, component connections, failure/recovery behavior, essential data and training traits.'),
('Disclosure and inventorship','Human conception/contribution log; earlier publications/presentations/repository visibility; college/owner review.'),
('Decision','Continue, revise, engineering-only, paper/capstone-only or patent-assessment packet; reason and next review trigger.')]:
    parts.append(f'**{label}.** {desc}\n')
parts.append('## 19. Completion and limits of this audit\n\nAll 385 owned Python files are in the ledger, all source roots are in the module matrix, all 24 research families have an experiment and disposition, and all 56 highlights are mapped to original-guideline pages. Code hashes were checked against the structural inventory before authoring. The saved guide and original PDF were not edited.\n\nThis is a deep research foundation, not a proven invention selection. Exhaustive worldwide prior art, patent-family/legal-status/FTO review, public-disclosure chronology, ownership determination, full current integration tests and new real-model/GPU benchmark campaigns remain uncompleted. The report states these limits where they affect decisions. The team should choose a narrow track using evidence and be willing to stop a patent angle when ordinary engineering or earlier work explains the result.\n')
parts.append('## 20. Source authority and official references\n\nOriginal supplied document: C:/Users/Yuv Nahar/Downloads/Indian_patent_guidelines_CRI-AIML.pdf (67 pages). User-annotated study guide: C:/PRIMA_next/output/pdf/Indian_CRI_AI_ML_Plain_Language_Guide.pdf (27 pages, 56 highlights). These source files are preserved.\n\nOfficial publication context: [IP India guidelines index](https://ipindia.gov.in/resource/patents-resources-guidelines) and [2025 CRI guidelines](https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf). This report applies the supplied document; it does not conflate it with later guidelines about using AI within patent examination.\n')
md='\n\n'.join(parts).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2019',"'")
md=re.sub(r'(?m)^(\|.*)\n\n(?=\|)',r'\1\n',md)
md_path=OUT/(BASE+'.md');md_path.write_text(md,encoding='utf-8')

# A small Markdown subset renders the same authored content into a paginated PDF.
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
styles={
 'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#124b53'),spaceAfter=22),
 'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=14,leading=18,textColor=colors.HexColor('#124b53'),spaceBefore=14,spaceAfter=9,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='Arial-Bold',fontSize=10.7,leading=14,textColor=colors.HexColor('#124b53'),spaceBefore=11,spaceAfter=5,keepWithNext=True),
 'body':ParagraphStyle('body',fontName='Arial',fontSize=9.3,leading=13.1,spaceAfter=8,splitLongWords=True),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=7.8,leading=10.5,spaceAfter=0),
 'headcell':ParagraphStyle('headcell',fontName='Arial-Bold',fontSize=7.6,leading=10,textColor=colors.white),
 'foot':ParagraphStyle('foot',fontName='Arial',fontSize=7,leading=9,textColor=colors.HexColor('#617079'))}
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\(&lt;([^&]+)&gt;\)',lambda m:f'<link href="{m[2]}">{m[1]}</link>',s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',lambda m:f'<link href="{m[2]}" color="#155c73">{m[1]}</link>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'\1',s)
    return s
def para(s,sty='body'):return Paragraph(inline(s),styles[sty])
story=[]
lines=md.splitlines();i=0
while i<len(lines):
    s=lines[i].strip()
    if not s:i+=1;continue
    if s=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
    if s.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells): table.append(cells)
            i+=1
        n=len(table[0]); widths={6:[35,177,25,71,69,66],4:[77,62,79,225]}[n]
        if table[0][0].startswith('Highlight'):widths=[62,103,55,223]
        data=[[Paragraph(f'<link href="#{c}" color="#155c73">{c}</link>',styles['cell']) if j>0 and re.fullmatch(r'R\d\d',c) else para(c,'headcell' if j==0 else 'cell') for c in row] for j,row in enumerate(table)]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#124b53')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f6f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#124b53'))]))
        story.extend([t,Spacer(1,12)]);continue
    if s.startswith('### '):story.append(para(s[4:],'h3'));i+=1;continue
    if s.startswith('## '):story.append(para(s[3:],'h2'));i+=1;continue
    if s.startswith('# '):story.append(para(s[2:],'title'));i+=1;continue
    block=[s];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','<!--')):
        block.append(lines[i].strip());i+=1
    story.append(para(' '.join(block)))
class Report(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name in {'h2','h3'}:
            key=f'outline_{self.page}_{len(self.canv._code)}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(),key,0 if f.style.name=='h2' else 1,False)
            rid=re.match(r'(R\d\d)\.',f.getPlainText())
            if rid:self.canv.bookmarkPage(rid[1])
def footer(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(colors.HexColor('#c5d6d7'));canvas.line(62,43,w-62,43)
    canvas.setFont('Arial',7);canvas.setFillColor(colors.HexColor('#617079'))
    canvas.drawString(62,31,'PRIMA-NEXT | Research foundation | 16 Oct 2026 | Not a patentability determination')
    canvas.drawRightString(w-62,31,str(doc.page));canvas.restoreState()
pdf_path=OUT/(BASE+'.pdf')
doc=Report(str(pdf_path),pagesize=A4,rightMargin=74,leftMargin=74,topMargin=56,bottomMargin=58,title='PRIMA-NEXT Patent Research Foundation',author='PRIMA research audit',allowSplitting=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(pdf_path)
text='\n'.join(p.extract_text() or '' for p in reader.pages)
assert len(reader.pages)>20
for rid in byid:assert f'{rid}.' in text,rid
for h in highlights:assert h['id'] in text,h['id']
assert len([r for r in rows if r['asset_type']=='owned_python'])==385
assert '385' in text and '44,720' in text
manifest={'date':'2026-10-16','head':inv['head'],'owned_python_files':385,'owned_python_lines':44720,'owned_definitions':2735,'ledger_rows':len(rows),'non_python_assets':len([r for r in rows if r['asset_type']=='non_python_asset']),'third_party_tree_boundaries':3,'research_families':24,'highlights_mapped':56,'pdf_pages':len(reader.pages),'words_markdown':len(md.split()),'outputs':[str(p) for p in [pdf_path,md_path,rank_path,inventory_path]],'validation':'All Python SHA-256 checked; 24 IDs/56 highlights present; anchors exist; ledger Python coverage exact. Visual QA pending.'}
(TMP/'artifact_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
