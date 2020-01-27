#!/usr/bin/env python
import json
from collections import defaultdict
import random

# tables and jsons from
# https://twiki.cern.ch/twiki/bin/view/CMS/PdmV2017Analysis
# https://twiki.cern.ch/twiki/bin/view/CMS/PdmV2018Analysis

# ERA     Absolute Run Number     Collision Runs Only     Energy  Dataset name    Json Lumi (/fb) Json Lumi (/fb) Certification Notes
# Run2017B        297020  299329  297046  299329  13      /PromptReco/Collisions2017B/DQM 4.792   4.823
# Run2017C        299337  302029  299368  302029  13      /PromptReco/Collisions2017C/DQM 9.755   9.664
# Run2017D        302030  303434  302030  303434  13      /PromptReco/Collisions2017D/DQM 4.319   4.252
# Run2017E        303435  304826  303824  304797  13      /PromptReco/Collisions2017E/DQM 9.424   9.278
# Run2017F        304911  306462  305040  306462  13      /PromptReco/Collisions2017F/DQM 13.50   13.540
# Run2018A        315252  316995  315252  316995  13      /PromptReco/Collisions2018A/DQM 13.48 /fb       14.00 /fb
# Run2018B        316998  319312  317080  319310  13      /PromptReco/Collisions2018B/DQM 6.785 /fb       7.10 /fb
# Run2018C        319313  320393  319337  320065  13      /PromptReco/Collisions2018C/DQM 6.612 /fb       6.94 /fb
# Run2018D        320394  325273  320673  325175  13      /PromptReco/Collisions2018D/DQM 31.95 /fb       31.93 /fb

def get_era(run):
    run = int(run)
    if 297020 <= run <= 299329: return "2017B"
    if 299337 <= run <= 302029: return "2017C"
    if 302030 <= run <= 303434: return "2017D"
    if 303435 <= run <= 304826: return "2017E"
    if 304911 <= run <= 306462: return "2017F"
    if 315252 <= run <= 316995: return "2018A"
    if 316998 <= run <= 319312: return "2018B"
    if 319313 <= run <= 320393: return "2018C"
    if 320394 <= run <= 325273: return "2018D"
    return "Unknown"

lumis = dict()
lumis.update(json.load(open("Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt")))
lumis.update(json.load(open("Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt")))

eras_to_runs = defaultdict(list)
for k,v in lumis.items():
    era = get_era(k)
    eras_to_runs[era].append(k)

# for each era, get ~10% of randomly chosen runs from
# the old golden json and stick it into the new one
random.seed(44)
newlumis = dict()
for era,runs in sorted(eras_to_runs.items()):
    newruns = random.sample(runs, int(round(len(runs)/10)))
    print("{}: {} -> {} runs".format(era, len(runs), len(newruns)))
    for run in newruns:
        newlumis[run] = lumis[run]
outname = "Cert_2017-2018_10percentbyrun_JSON.txt"
with open(outname, "w") as fh:
    json.dump(newlumis, fh)
print("Made {}".format(outname))



