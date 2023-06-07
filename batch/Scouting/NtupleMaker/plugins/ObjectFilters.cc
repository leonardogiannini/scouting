#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"

typedef ObjectCountFilter<Run3ScoutingMuonCollection>::type ScoutingMuonCountFilter;
DEFINE_FWK_MODULE(ScoutingMuonCountFilter);
typedef ObjectCountFilter<Run3ScoutingVertexCollection>::type ScoutingVertexCountFilter;
DEFINE_FWK_MODULE(ScoutingVertexCountFilter);
