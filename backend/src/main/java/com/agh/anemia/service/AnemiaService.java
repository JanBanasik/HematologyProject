package com.agh.anemia.service;

import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.PredictionResponse;
import com.agh.anemia.repository.BloodTestResultRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
public class AnemiaService {
    @Autowired
    private BloodTestResultRepository repository;

    private final RestTemplate restTemplate = new RestTemplate();
    private final String fastApiUrl = "http://localhost:8000/predict";

    public BloodTestResult predict(BloodTestResult input) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<BloodTestResult> request = new HttpEntity<>(input, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(fastApiUrl, request, Map.class);
        Map body = response.getBody();

        input.setPrediction((String) body.get("prediction"));
        input.setProbability(((Number) body.get("probability")).doubleValue());

        return repository.save(input);
    }

    public Iterable<BloodTestResult> getAll() {
        return repository.findAll();
    }

    public BloodTestResult predictAndSave(BloodTestResult result) {
        PredictionResponse resp = callFastApi(result);
        result.setPrediction(resp.getPrediction());
        result.setProbability(resp.getProbability());
        return repository.save(result);
    }

    public PredictionResponse callFastApi(BloodTestResult input) {
        RestTemplate restTemplate = new RestTemplate();
        String url = "http://localhost:8000/predict";

        // utwórz mapę z danymi wejściowymi
        Map<String, Double> body = Map.of(
                "RBC", input.getRBC(),
                "HGB", input.getHGB(),
                "HCT", input.getHCT(),
                "MCV", input.getMCV(),
                "MCH", input.getMCH(),
                "MCHC", input.getMCHC(),
                "RDW", input.getRDW(),
                "PLT", input.getPLT(),
                "WBC", input.getWBC()
        );

        // wywołanie FastAPI
        ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                url,
                body,
                PredictionResponse.class
        );

        return response.getBody();
    }

    /** Tylko zapis – bez wywoływania FastAPI */
    public BloodTestResult save(BloodTestResult result) {
        return repository.save(result);
    }


}
