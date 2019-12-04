#!/usr/bin/env bash

tar cvz psets/2018/*.py gridpacks/gridpack.tar.gz -C ../batch/ "Scouting/NtupleMaker/plugins/" "Scouting/NtupleMaker/src/" -f package.tar.gz

echo
echo "Did you remember to copy the right pset over psets/2018/slimmer_cfg.py first?"
echo

# output structure (`tar tvf package.tar.gz`) will be 
#   psets/2018/aodsim_cfg.py
#   psets/2018/gensim_cfg.py
#   psets/2018/miniaodsim_cfg.py
#   psets/2018/rawsim_cfg.py
#   psets/2018/slimmer_cfg.py
#   gridpacks/gridpack.tar.gz
#   Scouting/NtupleMaker/plugins/
#   Scouting/NtupleMaker/plugins/BuildFile.xml
#   Scouting/NtupleMaker/plugins/ObjectFilters.cc
#   Scouting/NtupleMaker/plugins/TriggerMaker.cc
#   Scouting/NtupleMaker/plugins/TriggerMaker.h

