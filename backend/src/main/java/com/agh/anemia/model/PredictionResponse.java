package com.agh.anemia.model;

public class PredictionResponse {
    private String prediction;
    private String probabilityLabel;


    public String getProbabilityLabel() {
        return probabilityLabel;
    }

    public void setProbabilityLabel(String probabilityLabel) {
        this.probabilityLabel = probabilityLabel;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }
}
