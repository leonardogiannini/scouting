#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"

typedef ObjectCountFilter<ScoutingMuonCollection>::type ScoutingMuonCountFilter;
DEFINE_FWK_MODULE(ScoutingMuonCountFilter);
typedef ObjectCountFilter<ScoutingVertexCollection>::type ScoutingVertexCountFilter;
DEFINE_FWK_MODULE(ScoutingVertexCountFilter);
