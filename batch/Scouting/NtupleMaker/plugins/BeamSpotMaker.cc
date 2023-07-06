#include "Scouting/NtupleMaker/plugins/BeamSpotMaker.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TMath.h"

using namespace edm;
using namespace std;

BeamSpotMaker::BeamSpotMaker(const edm::ParameterSet& iConfig) {

  beamSpotToken = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));

  produces<float>("x").setBranchAlias("x");
  produces<float>("y").setBranchAlias("y");
  produces<float>("z").setBranchAlias("z");
  produces<float>("dxdz").setBranchAlias("dxdz");
  produces<float>("dydz").setBranchAlias("dydz");
}

BeamSpotMaker::~BeamSpotMaker(){}

void BeamSpotMaker::beginJob(){}

void BeamSpotMaker::endJob(){}

void BeamSpotMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  unique_ptr<float> x(new float);
  unique_ptr<float> y(new float);
  unique_ptr<float> z(new float);
  unique_ptr<float> dxdz(new float);
  unique_ptr<float> dydz(new float);

  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(beamSpotToken, beamSpotH);

  // bool haveBeamSpot = true;
  // if(!beamSpotH.isValid() )
  //   haveBeamSpot = false;

  *x = beamSpotH->position().x();
  *y = beamSpotH->position().y();
  *z = beamSpotH->position().z();
  *dxdz = beamSpotH->dxdz();
  *dydz = beamSpotH->dydz();

  iEvent.put(std::move(x), "x");
  iEvent.put(std::move(y), "y");
  iEvent.put(std::move(z), "z");
  iEvent.put(std::move(dxdz), "dxdz");
  iEvent.put(std::move(dydz), "dydz");
  
}

DEFINE_FWK_MODULE(BeamSpotMaker);
