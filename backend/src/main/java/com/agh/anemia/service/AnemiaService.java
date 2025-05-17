// src/main/java/com/agh/anemia/service/AnemiaService.java
package com.agh.anemia.service;

import com.agh.anemia.dto.BloodTestPredictionDto;
import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.PredictionResponse; // Prawdopodobnie używane tylko w callFastApi
import com.agh.anemia.model.User;
import com.agh.anemia.repository.BloodTestResultRepository;
import org.springframework.http.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails; // Importuj klasy Spring Security
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional; // Nadal potrzebne dla metod zapisu
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.Map;

// ... (klasa UserAlreadyExistsException, jeśli jest zdefiniowana w tym pliku) ...


@Service
@Transactional // Adnotacja na poziomie klasy - domyślnie metody publiczne są transakcyjne
public class AnemiaService {

    // Repozytorium do interakcji z bazą danych dla BloodTestResult
    private final BloodTestResultRepository repository;

    // RestTemplate do wywoływania zewnętrznych usług (FastAPI)
    private final RestTemplate restTemplate;

    // URL endpointu FastAPI
    private final String fastApiUrl = "http://localhost:8000/predict";


    // Wstrzyknij repozytorium przez konstruktor (zalecane)
    public AnemiaService(BloodTestResultRepository repository) {
        this.repository = repository;
        this.restTemplate = new RestTemplate();
    }

    /**
     * Metoda pomocnicza do pobierania obecnie zalogowanego użytkownika ze Spring Security Context.
     *
     * @return Zalogowany obiekt User lub null, jeśli użytkownik nie jest zalogowany lub nie jest typu User.
     */
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


    /**
     * Obsługuje predykcję i zapis wyniku po przesłaniu formularza Thymeleaf do /anemia/predict.
     * Otrzymuje encję BloodTestResult bezpośrednio z formularza.
     * Ta metoda jest transakcyjna ze względu na zapis.
     *
     * @param result Encja BloodTestResult z danymi wejściowymi (mapowana z @ModelAttribute).
     * @return Zapisana encja BloodTestResult z dodanymi wynikami predykcji.
     */
    // Metoda jest domyślnie transakcyjna dzięki adnotacji na poziomie klasy @Transactional
    // @Transactional // Możesz jawnie dodać, jeśli chcesz nadpisać domyślne ustawienia klasy
    public BloodTestResult predictAndSave(BloodTestResult result) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            throw new IllegalStateException("Użytkownik nie jest zalogowany, a próbuje wykonać predykcję!");
        }
        result.setUser(currentUser);

        PredictionResponse resp = callFastApi(result);
        result.setPrediction(resp.getPrediction());
        result.setProbability(resp.getProbability());
        // result.setEpicrisis(resp.getEpicrisis());

        System.out.println("Saving BloodTestResult entity received from form/modelAttribute after FastAPI call: " + result.toString());

        return repository.save(result); // Zapisz encję
    }

    /**
     * Pobiera wyniki predykcji TYLKO dla zalogowanego użytkownika.
     * Używane przez kontroler do wyświetlania historii.
     * Jest to operacja TYLKO DO ODCZYTU, NIE POWINNA BYĆ TRANSAKCYJNA.
     *
     * @return Lista BloodTestResult dla zalogowanego użytkownika.
     */
    // USUNIĘTO JAWNĄ ADNOTACJĘ @Transactional, jeśli tam była
    // Ponieważ @Transactional jest na poziomie klasy, domyślnie metody publiczne SĄ transakcyjne.
    // ABY ZMIENIĆ TO ZACHOWANIE DLA TEJ METODY, MUSIMY JĄ OZNACZYĆ @Transactional(readOnly = true) LUB PODOBNIE,
    // ALBO USUNĄĆ @Transactional z poziomu klasy i dodawać ją tylko do metod zapisu.
    // Dla prostoty, spróbujmy jawnie oznaczyć jako readOnly, jeśli to @Transactional na klasie powoduje problem
    @Transactional(readOnly = true) // Oznacz jawnie jako tylko do odczytu
    public Iterable<BloodTestResult> getHistoryForCurrentUser() {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            return Collections.emptyList();
        }
        // Ta metoda jest teraz jawnie oznaczona jako tylko do odczytu
        return repository.findByUser(currentUser);
    }

    /**
     * Wywołuje zewnętrzną usługę FastAPI do predykcji.
     *
     * @param input Obiekt z danymi wejściowymi do predykcji.
     * @return Obiekt PredictionResponse z wynikiem i pewnością.
     */
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


    /**
     * Zapisuje wynik predykcji otrzymany jako DTO z frontendu (po predykcji JS fetch).
     * Tworzy nową encję BloodTestResult na podstawie DTO i zapisuje ją w bazie.
     * Ta metoda jest transakcyjna ze względu na zapis.
     *
     * @param resultDto DTO z danymi wejściowymi i wynikami predykcji otrzymane z frontendu.
     * @return Zapisana encja BloodTestResult.
     */
    // Metoda jest domyślnie transakcyjna dzięki adnotacji na poziomie klasy @Transactional
    // @Transactional // Możesz jawnie dodać, jeśli chcesz nadpisać domyślne ustawienia klasy
    public BloodTestResult savePredictionResult(BloodTestPredictionDto resultDto) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            throw new IllegalStateException("Użytkownik nie jest zalogowany, a próbuje zapisać wynik!");
        }

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
        result.setProbability(resultDto.getProbability());
        result.setEpicrisis(resultDto.getEpicrisis());

        result.setUser(currentUser);
        // result.onCreate(); // @PrePersist handles this when saving

        System.out.println("Saving BloodTestResult entity created from DTO: " + result.toString());

        return repository.save(result);
    }

}