source /cvmfs/cms.cern.ch/cmsset_default.sh

thisdir=$(pwd)
release=CMSSW_10_2_5
cmsrel $release
cd $_
cmsenv
cd -

git cms-addpkg RecoTracker/TkDetLayers
patch $CMSSW_BASE/src/RecoTracker/TkDetLayers/src/TBLayer.cc < $thisdir/patches/RecoTracker_TkDetLayers_src_TBLayer.patch

cd $release/src
ln -s ../../Scouting/
scram b -j10
cd -
