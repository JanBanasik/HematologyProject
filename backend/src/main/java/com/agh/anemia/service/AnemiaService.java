package com.agh.anemia.service;

import com.agh.anemia.dto.BloodTestPredictionDto;
import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.PredictionResponse;
import com.agh.anemia.model.User;
import com.agh.anemia.repository.BloodTestResultRepository;
import org.springframework.http.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;


@Service
@Transactional
public class AnemiaService {

    private final BloodTestResultRepository repository;

    private final RestTemplate restTemplate;

    private final String fastApiUrl = "http://localhost:8000/predict";


    public AnemiaService(BloodTestResultRepository repository) {
        this.repository = repository;
        this.restTemplate = new RestTemplate();
    }

    private User getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        if (authentication != null && authentication.isAuthenticated() && !(authentication.getPrincipal() instanceof String)) {
            Object principal = authentication.getPrincipal();
            if (principal instanceof UserDetails) {
                return (User) principal;
            }
        }
        return null;
    }

    public BloodTestResult getResultByIdAndUser(Long id) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            return null;
        }

        Optional<BloodTestResult> resultOptional = repository.findById(id);

        if (resultOptional.isPresent() && resultOptional.get().getUser().getId().equals(currentUser.getId())) {
            return resultOptional.get();
        } else {
            return null;
        }
    }


    public BloodTestResult predictAndSave(BloodTestResult result) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            throw new IllegalStateException("Użytkownik nie jest zalogowany, a próbuje wykonać predykcję!");
        }
        result.setUser(currentUser);

        PredictionResponse resp = callFastApi(result);
        result.setPrediction(resp.getPrediction());
        result.setProbabilityLabel(resp.getProbabilityLabel());

        System.out.println("Saving BloodTestResult entity received from form/modelAttribute after FastAPI call: " + result.toString());

        return repository.save(result);
    }

    public Iterable<BloodTestResult> getHistoryForCurrentUser() {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            return Collections.emptyList();
        }
        return repository.findByUser(currentUser);
    }

    public PredictionResponse callFastApi(BloodTestResult input) {
        String url = "http://localhost:8000/predict";

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

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Double>> request = new HttpEntity<>(body, headers);

        ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                url,
                request,
                PredictionResponse.class
        );

        return response.getBody();
    }


    public BloodTestResult savePredictionResult(BloodTestPredictionDto resultDto) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            throw new IllegalStateException("Użytkownik nie jest zalogowany, a próbuje zapisać wynik!");
        }
        System.out.println("Received DTO getProbabilityLabel(): " + resultDto.getProbabilityLabel()); // Dodaj ten log

        BloodTestResult result = new BloodTestResult();

        result.setRBC(resultDto.getRBC());
        result.setHGB(resultDto.getHGB());
        result.setHCT(resultDto.getHCT());
        result.setMCV(resultDto.getMCV());
        result.setMCH(resultDto.getMCH());
        result.setMCHC(resultDto.getMCHC());
        result.setRDW(resultDto.getRDW());
        result.setPLT(resultDto.getPLT());
        result.setWBC(resultDto.getWBC());
        result.setPrediction(resultDto.getPrediction());
        result.setProbabilityLabel(resultDto.getProbabilityLabel());
        result.setEpicrisis(resultDto.getEpicrisis());

        result.setUser(currentUser);
        String probabilityLabelFromDto = resultDto.getProbabilityLabel();
        System.out.println("Value from DTO before setting in entity: " + probabilityLabelFromDto); // Dodaj ten log
        System.out.println("Value in entity after setting: " + result.getProbabilityLabel()); // Dodaj ten log
        System.out.println("Saving BloodTestResult entity created from DTO: " + result);

        return repository.save(result);
    }

}