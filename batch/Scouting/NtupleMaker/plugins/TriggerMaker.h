#ifndef NTUPLEMAKER_TRIGGERMAKER_H
#define NTUPLEMAKER_TRIGGERMAKER_H

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include <memory>

#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

// system include files
#include <algorithm>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "FWCore/Common/interface/TriggerNames.h"


class TriggerMaker : public edm::stream::EDProducer<> {
public:
  explicit TriggerMaker(const edm::ParameterSet&);
  ~TriggerMaker();

private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  bool doL1_;
  bool doTriggerObjects_;
  triggerExpression::Data triggerCache_;
  std::vector<triggerExpression::Evaluator*> vtriggerSelector_;
  std::vector<std::string> vtriggerAlias_, vtriggerSelection_;

  edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjectsToken;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken;
  // edm::EDGetTokenT<pat::PackedTriggerPrescales> triggerPrescaleToken;

  edm::Handle<edm::TriggerResults> triggerResultsH_;
  edm::Handle<trigger::TriggerEvent> triggerEventH_; 
  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjectStandAlonesH_;
  edm::TriggerNames triggerNames_;

  edm::EDGetToken algToken_;
  std::shared_ptr<l1t::L1TGlobalUtil> l1GtUtils_;
  std::vector<std::string> l1Seeds_;

};

#endif

