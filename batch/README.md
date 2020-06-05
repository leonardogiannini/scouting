## Info

RAW ntuples are generated/slimmed first, then these are made into babies.

### (Slimmed) RAW ntuples

#### Data

RAW scouting data is already available, but it stores huge FED and track info, so we first skim it with CRAB to events with
>=2 Scouting Muons and >=1 DV, and we add some extra information, while preserving the EDM structure. So in principle, one can run the same
CMSSW code on these new ntuples to add/compute other stuff.
* Do `. install_cmssw.sh`. This sets up a CMSSW environment with some EDFilters and some EDAnalyzers. (Note that `Scouting/NtupleMaker/test/producer.py` is the relevant PSet for slim/skimming)

##### Manual way
* Submit crab jobs
  * `source /cvmfs/cms.cern.ch/crab3/crab.sh` for crab commands
  * submit with
```bash
# A-D
crab submit -c crabcfg.py General.requestName="skim_2018A_v4" Data.inputDataset="/ScoutingCaloMuon/Run2018A-v1/RAW" ;
crab submit -c crabcfg.py General.requestName="skim_2018B_v4" Data.inputDataset="/ScoutingCaloMuon/Run2018B-v1/RAW" ;
crab submit -c crabcfg.py General.requestName="skim_2018C_v4" Data.inputDataset="/ScoutingCaloMuon/Run2018C-v1/RAW" ;
crab submit -c crabcfg.py General.requestName="skim_2018D_v4" Data.inputDataset="/ScoutingCaloMuon/Run2018D-v1/RAW" ;

# C again, but just the unblinded set
crab submit -c crabcfg.py General.requestName="skim_2018D_v4_unblind1fb" Data.inputDataset="/ScoutingCaloMuon/Run2018D-v1/RAW" Data.lumiMask="data/unblind_2018C_1fb_JSON.txt" Data.unitsPerJob=2000000;
```
  * Spam `crab status -c crab/<requestname>`, `crab resubmit -c ...`

##### Nicer way
* Submit crab jobs
  * `cmsenv` in the checked out release directory
  * `source /cvmfs/cms.cern.ch/crab3/crab.sh` for crab commands
  * Edit `multicrab.py` and `python multicrab.py` to submit
  * It also gives you a nice monitoring page (see the bottom of the script)

#### MC
* First submit jobs to generate and make RAW/scouting-tier data in the [generation folder](../generation/)
* Once those are done, proceed to the section on making babies below.


### Flattened babies (non-EDM)

* Run `python babymaker.py -h` to see all the options. A list of files can be specified. Note that the `-a` option will retain all events:
no skim of >=1 DV and >=2 Muon is performed (useful for MC samples where acceptance needs to be calculated).

* Clone [ProjectMetis](https://github.com/aminnj/ProjectMetis/) and source its environment
* Run the babymaker on a file locally to test
* Edit `submit_baby_jobs.py` to have the right paths (search for "hadoop") to the crab output, including the request names, output tags, and skimming options (search for "skim")
* `./make_tar.sh` to make the tarball for the jobs
* `python submit_baby_jobs.py`
