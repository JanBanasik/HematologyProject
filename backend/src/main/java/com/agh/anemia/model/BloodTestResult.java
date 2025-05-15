package com.agh.anemia.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
public class BloodTestResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

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


    private String prediction;


    private Double probability;

    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }


    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

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

    public Double getMCV() {
        return MCV;
    }

    public void setMCV(Double MCV) {
        this.MCV = MCV;
    }

    public Double getHCT() {
        return HCT;
    }

    public void setHCT(Double HCT) {
        this.HCT = HCT;
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

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    @Override
    public String toString() {
        return "BloodTestResult{" +
                "id=" + id +
                ", RBC=" + RBC +
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
                ", createdAt=" + createdAt +
                '}';
    }
}