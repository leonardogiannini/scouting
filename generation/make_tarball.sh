#!/usr/bin/env bash

tar cvz psets/201*/*.py gridpacks/gridpack.tar.gz -C ../batch/ "Scouting/NtupleMaker/plugins/" "Scouting/NtupleMaker/src/" -f package.tar.gz

echo
echo "Did you remember to copy the right pset over slimmer_cfg.py into psets/ first?"
echo

# output structure (`tar tvf package.tar.gz`) will be 
#   psets/2017/aodsim_cfg.py
#   psets/2017/gensim_cfg.py
#   psets/2017/rawsim_cfg.py
#   psets/2017/slimmer_cfg.py
#   psets/2018/aodsim_cfg.py
#   psets/2018/gensim_cfg.py
#   psets/2018/rawsim_cfg.py
#   psets/2018/slimmer_cfg.py
#   gridpacks/gridpack.tar.gz
#   Scouting/NtupleMaker/plugins/
#   Scouting/NtupleMaker/plugins/BuildFile.xml
#   Scouting/NtupleMaker/plugins/ObjectFilters.cc
#   Scouting/NtupleMaker/plugins/TriggerMaker.cc
#   Scouting/NtupleMaker/plugins/TriggerMaker.h

