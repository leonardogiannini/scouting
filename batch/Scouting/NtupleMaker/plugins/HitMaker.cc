#include "Scouting/NtupleMaker/plugins/HitMaker.h" 
#include "TMath.h"

using namespace edm;
using namespace std;

namespace {

  Surface::RotationType rotation(const GlobalVector& zDir) {
    GlobalVector zAxis = zDir.unit();
    GlobalVector yAxis(zAxis.y(), -zAxis.x(), 0);
    GlobalVector xAxis = yAxis.cross(zAxis);
    return Surface::RotationType(xAxis, yAxis, zAxis);
  }


}

HitMaker::HitMaker(const edm::ParameterSet& iConfig)
{
    muonToken_ = consumes<ScoutingMuonCollection>(iConfig.getParameter<InputTag>("muonInputTag"));
    dvToken_ = consumes<ScoutingVertexCollection>(iConfig.getParameter<InputTag>("dvInputTag"));
    measurementTrackerEventToken_ = consumes<MeasurementTrackerEvent>(iConfig.getParameter<InputTag>("measurementTrackerEventInputTag"));

    produces<vector<vector<bool> > >("isbarrel").setBranchAlias("Muon_hit_barrel");
    produces<vector<vector<bool> > >("isactive").setBranchAlias("Muon_hit_active");
    produces<vector<vector<int> > >("layernum").setBranchAlias("Muon_hit_layer");
    produces<vector<vector<int> > >("ndet").setBranchAlias("Muon_hit_ndet");
    produces<vector<vector<float> > >("x").setBranchAlias("Muon_hit_x");
    produces<vector<vector<float> > >("y").setBranchAlias("Muon_hit_y");
    produces<vector<vector<float> > >("z").setBranchAlias("Muon_hit_z");
    produces<vector<int> >("nexpectedhits").setBranchAlias("Muon_nExpectedPixelHits");
}

HitMaker::~HitMaker(){
}

void HitMaker::beginJob(){}

void HitMaker::endJob(){}

void HitMaker::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
    iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorHandle_);
    iSetup.get<GlobalTrackingGeometryRecord>().get(theGeo_);
    iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
    iSetup.get<CkfComponentsRecord>().get("", measurementTracker_);
}

void HitMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

    bool debug = true;

    auto const& searchGeom = *(*measurementTracker_).geometricSearchTracker();
    auto const& prop = *propagatorHandle_;

    edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
    iEvent.getByToken(measurementTrackerEventToken_, measurementTrackerEvent);

    edm::Handle<ScoutingMuonCollection> muonHandle;
    iEvent.getByToken(muonToken_, muonHandle);

    edm::Handle<ScoutingVertexCollection> dvHandle;
    iEvent.getByToken(dvToken_, dvHandle);

    unique_ptr<vector<vector<bool> > > v_isbarrel(new vector<vector<bool> >);
    unique_ptr<vector<vector<bool> > > v_isactive(new vector<vector<bool> >);
    unique_ptr<vector<vector<int> > > v_layernum(new vector<vector<int> >);
    unique_ptr<vector<vector<int> > > v_ndet(new vector<vector<int> >);
    unique_ptr<vector<vector<float> > > v_hitx(new vector<vector<float> >);
    unique_ptr<vector<vector<float> > > v_hity(new vector<vector<float> >);
    unique_ptr<vector<vector<float> > > v_hitz(new vector<vector<float> >);
    unique_ptr<vector<int> > v_nexpectedhits(new vector<int>);

    for (auto const& muon : *muonHandle) {
        vector<int> vertex_indices = muon.vtxIndx();
        int first_good_index = 0;
        for (auto idx : vertex_indices) {
            if (idx >= 0) {
                first_good_index = idx;
                break;
            }
        }
        int nDV = (*dvHandle).size();
        float dv_x = 0;
        float dv_y = 0;
        float dv_z = 0;
        if (first_good_index < nDV) {
            ScoutingVertex dv = (*dvHandle).at(first_good_index);
            dv_x = dv.x();
            dv_y = dv.y();
            dv_z = dv.z();
        }
        TLorentzVector lv;
        lv.SetPtEtaPhiM(muon.pt(), muon.eta(), muon.phi(), muon.phi());

        float track_px = lv.Px();
        float track_py = lv.Py();
        float track_pz = lv.Pz();
        float track_qoverpError = muon.trk_qoverpError();
        float track_lambdaError = muon.trk_lambdaError();
        float track_phiError = muon.trk_phiError();
        float track_dxyError = muon.dxyError();
        float track_dszError = muon.trk_dszError();
        int track_charge = muon.charge();
        int nvalidpixelhits = muon.nValidPixelHits();

        reco::TrackBase::CovarianceMatrix track_cov;
        track_cov(0,0) = pow(track_qoverpError,2);
        track_cov(1,1) = pow(track_lambdaError,2);
        track_cov(2,2) = pow(track_phiError,2);
        track_cov(3,3) = pow(track_dxyError,2);
        track_cov(4,4) = pow(track_dszError,2);
        CurvilinearTrajectoryError err(track_cov);
        // Default parameters according to https://github.com/cms-sw/cmssw/blob/master/TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorParams.h
        Chi2MeasurementEstimator estimator(30., 3.0, 0.5, 2.0, 0.5, 1.e12);

        GlobalVector startingMomentum(track_px, track_py, track_pz);
        GlobalPoint startingPosition(dv_x, dv_y, dv_z);

        PlaneBuilder pb;
        auto rot = rotation(startingMomentum);
        auto startingPlane = pb.plane(startingPosition, rot);

        auto mom = startingMomentum;
        TrajectoryStateOnSurface startingStateP(
                GlobalTrajectoryParameters(startingPosition, mom, track_charge, magfield_.product()),
                err, *startingPlane
                );

        // or could get searchGeom.allLayers() and require layer->subDetector() enum is PixelBarrel/PixelEndcap 
        vector<DetLayer const*> layers_pixel;
        for (auto layer : searchGeom.pixelBarrelLayers()) layers_pixel.push_back(layer);
        for (auto layer : searchGeom.negPixelForwardLayers()) layers_pixel.push_back(layer);
        for (auto layer : searchGeom.posPixelForwardLayers()) layers_pixel.push_back(layer);

        vector<bool> isbarrel;
        vector<bool> isactive;
        vector<int> layernum;
        vector<int> ndet;
        vector<float> hitx;
        vector<float> hity;
        vector<float> hitz;
        int nexpectedhits = 0;
        auto tsos = startingStateP;
        for (auto const& layer : layers_pixel) {
            // auto tsos = startingStateP;
            auto const& detWithState = layer->compatibleDets(tsos, prop, estimator);
            if (!detWithState.size()) continue;
            tsos = detWithState.front().second;
            DetId did = detWithState.front().first->geographicalId();
            MeasurementDetWithData measDet = measurementTracker_->idToDet(did, *measurementTrackerEvent);
            bool active = measDet.isActive();
            bool barrel = layer->isBarrel();
            int seq = layer->seqNum();
            int sdet = detWithState.size();
            auto pos = tsos.globalPosition();
            if (debug) {
                std::cout << "HIT subdet=" << layer->subDetector()
                    << " layer=" << seq << " detSize=" << sdet
                    << " pos=" << pos
                    << " active=" << active 
                    << std::endl;
            }
            isbarrel.push_back(barrel);
            isactive.push_back(active);
            layernum.push_back(seq);
            ndet.push_back(sdet);
            hitx.push_back(pos.x());
            hity.push_back(pos.y());
            hitz.push_back(pos.z());
            nexpectedhits += active;
        }
        v_isbarrel->push_back(isbarrel);
        v_isactive->push_back(isactive);
        v_layernum->push_back(layernum);
        v_ndet->push_back(ndet);
        v_hitx->push_back(hitx);
        v_hity->push_back(hity);
        v_hitz->push_back(hitz);
        v_nexpectedhits->push_back(nexpectedhits);

        if (debug) {
            std::cout <<  " nvalidpixelhits: " << nvalidpixelhits <<  " nexpectedhits: " << nexpectedhits <<  " nvalidpixelhits-nexpectedhits: " << nvalidpixelhits-nexpectedhits <<  std::endl;
        }

    }

    iEvent.put(std::move(v_isbarrel), "isbarrel");
    iEvent.put(std::move(v_isactive), "isactive");
    iEvent.put(std::move(v_layernum), "layernum");
    iEvent.put(std::move(v_ndet), "ndet");
    iEvent.put(std::move(v_hitx), "x");
    iEvent.put(std::move(v_hity), "y");
    iEvent.put(std::move(v_hitz), "z");
    iEvent.put(std::move(v_nexpectedhits), "nexpectedhits");
}

DEFINE_FWK_MODULE(HitMaker);
