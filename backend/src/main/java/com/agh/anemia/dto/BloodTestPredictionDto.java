// src/main/java/com/agh/anemia/dto/BloodTestPredictionDto.java
package com.agh.anemia.dto;

import com.fasterxml.jackson.annotation.JsonProperty; // Importuj adnotację Jacksona
import jakarta.persistence.Column;

// To jest obiekt, który Jackson będzie tworzył na podstawie JSON wysłanego z frontendu
// Nie ma adnotacji JPA (@Entity, @Id, etc.) - to prosty obiekt Java (POJO)
public class BloodTestPredictionDto {

    // Pola odpowiadające danym wejściowym z formularza, z adnotacjami JsonProperty,
    // aby Jackson poprawnie zmapował JSON, gdzie nazwy pól mogą być inne (np. wielkie litery)
    @JsonProperty("RBC")
    private Double RBC;
    @JsonProperty("HGB")
    private Double HGB;
    @JsonProperty("HCT")
    private Double HCT;
    @JsonProperty("MCV")
    private Double MCV;
    @JsonProperty("MCH")
    private Double MCH;
    @JsonProperty("MCHC")
    private Double MCHC;
    @JsonProperty("RDW")
    private Double RDW;
    @JsonProperty("PLT")
    private Double PLT;
    @JsonProperty("WBC")
    private Double WBC;

    // Pola odpowiadające wynikom predykcji otrzymanym z FastAPI i dodanym w JS
    private String prediction;
    private Double probability;

    @Column(columnDefinition="MEDIUMTEXT")
    private String epicrisis; // Dodaj to pole, jeśli FastAPI zwraca epikryzę


    // Domyślny konstruktor (wymagany przez Jacksona do deserializacji)
    public BloodTestPredictionDto() {
    }

    // Getters and Setters (potrzebne Jacksonowi do wypełnienia obiektu podczas deserializacji
    // oraz do uzyskania wartości w serwisie)

    public Double getRBC() {
        return RBC;
    }

    public void setRBC(Double RBC) {
        this.RBC = RBC;
    }

    public Double getHGB() {
        return HGB;
    }

    public void setHGB(Double HGB) {
        this.HGB = HGB;
    }

    public Double getHCT() {
        return HCT;
    }

    public void setHCT(Double HCT) {
        this.HCT = HCT;
    }

    public Double getMCV() {
        return MCV;
    }

    public void setMCV(Double MCV) {
        this.MCV = MCV;
    }

    public Double getMCH() {
        return MCH;
    }

    public void setMCH(Double MCH) {
        this.MCH = MCH;
    }

    public Double getMCHC() {
        return MCHC;
    }

    public void setMCHC(Double MCHC) {
        this.MCHC = MCHC;
    }

    public Double getRDW() {
        return RDW;
    }

    public void setRDW(Double RDW) {
        this.RDW = RDW;
    }

    public Double getPLT() {
        return PLT;
    }

    public void setPLT(Double PLT) {
        this.PLT = PLT;
    }

    public Double getWBC() {
        return WBC;
    }

    public void setWBC(Double WBC) {
        this.WBC = WBC;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public Double getProbability() {
        return probability;
    }

    public void setProbability(Double probability) {
        this.probability = probability;
    }

    public String getEpicrisis() {
        return epicrisis;
    }

    public void setEpicrisis(String epicrisis) {
        this.epicrisis = epicrisis;
    }


    @Override
    public String toString() {
        return "BloodTestPredictionDto{" +
                "RBC=" + RBC +
                ", HGB=" + HGB +
                ", HCT=" + HCT +
                ", MCV=" + MCV +
                ", MCH=" + MCH +
                ", MCHC=" + MCHC +
                ", RDW=" + RDW +
                ", PLT=" + PLT +
                ", WBC=" + WBC +
                ", prediction='" + prediction + '\'' +
                ", probability=" + probability +
                ", epicrisis='" + epicrisis + '\'' +
                '}';
    }
}