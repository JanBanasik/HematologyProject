// src/main/java/com/agh/anemia/model/BloodTestResult.java
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
    // Zakładając, że dodałeś pole epicrisis z FastAPI
    private String epicrisis;

    // DODAJ TO POLE DLA POWIĄZANIA Z UŻYTKOWNIKIEM
    @ManyToOne // Wiele BloodTestResult należy do jednego Usera
    @JoinColumn(name = "user_id") // Kolumna w tabeli blood_test_result, która będzie kluczem obcym do tabeli users
    private User user;


    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }

    // Getters and Setters (dla nowych pól też)

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Double getRBC() { return RBC; }
    public void setRBC(Double RBC) { this.RBC = RBC; }
    public Double getHGB() { return HGB; }
    public void setHGB(Double HGB) { this.HGB = HGB; }
    public Double getMCV() { return MCV; }
    public void setMCV(Double MCV) { this.MCV = MCV; }
    public Double getHCT() { return HCT; }
    public void setHCT(Double HCT) { this.HCT = HCT; }
    public Double getMCH() { return MCH; }
    public void setMCH(Double MCH) { this.MCH = MCH; }
    public Double getMCHC() { return MCHC; }
    public void setMCHC(Double MCHC) { this.MCHC = MCHC; }
    public Double getRDW() { return RDW; }
    public void setRDW(Double RDW) { this.RDW = RDW; }
    public Double getPLT() { return PLT; }
    public void setPLT(Double PLT) { this.PLT = PLT; }
    public Double getWBC() { return WBC; }
    public void setWBC(Double WBC) { this.WBC = WBC; }
    public String getPrediction() { return prediction; }
    public void setPrediction(String prediction) { this.prediction = prediction; }
    public Double getProbability() { return probability; }
    public void setProbability(Double probability) { this.probability = probability; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public String getEpicrisis() { return epicrisis; }
    public void setEpicrisis(String epicrisis) { this.epicrisis = epicrisis; }

    // DODAJ GETTER I SETTER DLA POLA USER
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }


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
                ", epicrisis='" + epicrisis + '\'' +
                ", user=" + (user != null ? user.getUsername() : "null") + // Pokaż nazwę użytkownika w stringu
                '}';
    }
}