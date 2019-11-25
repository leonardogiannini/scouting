#include "Scouting/NtupleMaker/plugins/TriggerMaker.h" 
#include "TMath.h"

using namespace edm;
using namespace std;

TriggerMaker::TriggerMaker(const edm::ParameterSet& iConfig) :
  doL1_(iConfig.getParameter<bool>("doL1")),
  triggerCache_(triggerExpression::Data(iConfig.getParameterSet("triggerConfiguration"), consumesCollector())),
  vtriggerAlias_(iConfig.getParameter<vector<string>>("triggerAlias")),
  vtriggerSelection_(iConfig.getParameter<vector<string>>("triggerSelection")),
  l1GtUtils_(nullptr)
{
  vtriggerSelector_.reserve(vtriggerSelection_.size());
  for (auto const& vt:vtriggerSelection_) vtriggerSelector_.push_back(triggerExpression::parse(vt));

  if (doL1_){
    algToken_ = consumes<BXVector<GlobalAlgBlk>>(iConfig.getParameter<InputTag>("AlgInputTag"));
    l1Seeds_ = iConfig.getParameter<std::vector<std::string> >("l1Seeds");
    l1GtUtils_ = std::make_shared<l1t::L1TGlobalUtil>(iConfig, consumesCollector());
  }

  produces<std::vector<std::string> >("l1name").setBranchAlias("l1_name");
  produces<std::vector<bool> >("l1result").setBranchAlias("l1_result");
  produces<std::vector<int> >("l1prescale").setBranchAlias("l1_prescale");

  produces<std::vector<std::string> >("hltname").setBranchAlias("hlt_name");
  produces<std::vector<bool> >("hltresult").setBranchAlias("hlt_result");
}

TriggerMaker::~TriggerMaker(){
  for (auto& vts:vtriggerSelector_) delete vts;
}

void TriggerMaker::beginJob(){}

void TriggerMaker::endJob(){}

void TriggerMaker::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){}

void TriggerMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  unique_ptr<vector<string> > l1_name(new vector<string>);
  unique_ptr<vector<bool> > l1_result(new vector<bool>);
  unique_ptr<vector<int> > l1_prescale(new vector<int>);

  unique_ptr<vector<string> > hlt_name(new vector<string>);
  unique_ptr<vector<bool> > hlt_result(new vector<bool>);

  if (triggerCache_.setEvent(iEvent, iSetup)){
    auto trigAlias=vtriggerAlias_.cbegin();
    for (auto& vts:vtriggerSelector_){
      bool result = false;
      if (vts){
        if (triggerCache_.configurationUpdated()) vts->init(triggerCache_);
        result = (*vts)(triggerCache_);
      }
      hlt_result->push_back(result);
      hlt_name->push_back(*trigAlias);
      trigAlias++;
    }
  }

  if (doL1_){
    l1GtUtils_->retrieveL1(iEvent, iSetup, algToken_);
    for (auto const& l1seed:l1Seeds_){
      bool l1htbit = 0;
      int prescale = -1;
      l1GtUtils_->getFinalDecisionByName(l1seed, l1htbit);
      l1GtUtils_->getPrescaleByName(l1seed, prescale);
      l1_result->push_back(l1htbit);
      l1_name->push_back(l1seed);
      l1_prescale->push_back(prescale);
    }
  }

  iEvent.put(std::move(l1_name), "l1name");
  iEvent.put(std::move(l1_result), "l1result");
  iEvent.put(std::move(l1_prescale), "l1prescale");
  iEvent.put(std::move(hlt_name), "hltname");
  iEvent.put(std::move(hlt_result), "hltresult");
}

DEFINE_FWK_MODULE(TriggerMaker);
