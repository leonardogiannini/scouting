#ifndef NTUPLEMAKER_BEAMSPOTMAKER_H
#define NTUPLEMAKER_BEAMSPOTMAKER_H

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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <memory>


class BeamSpotMaker : public edm::stream::EDProducer<> {
public:
  explicit BeamSpotMaker(const edm::ParameterSet&);
  ~BeamSpotMaker();

private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;

};

#endif

