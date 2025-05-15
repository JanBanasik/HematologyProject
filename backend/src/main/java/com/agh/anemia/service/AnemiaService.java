
package com.agh.anemia.service;

import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.PredictionResponse;
import com.agh.anemia.model.User;
import com.agh.anemia.repository.BloodTestResultRepository;
import org.springframework.http.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails; // Importuj klasy Spring Security
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Collections; // Dla pustej listy, jeśli trzeba obsłużyć przypadek braku usera
import java.util.Map;

@Service
public class AnemiaService {

    private final BloodTestResultRepository repository;
    // Nie potrzebujemy wstrzykiwać UserDetailsService tutaj, możemy pobrać usera z kontekstu bezpieczeństwa

    private final RestTemplate restTemplate; // Użyj final, inicjuj w konstruktorze
    private final String fastApiUrl = "http://localhost:8000/predict";

    // Wstrzyknij repozytorium przez konstruktor (zalecane)
    public AnemiaService(BloodTestResultRepository repository) {
        this.repository = repository;
        this.restTemplate = new RestTemplate(); // Inicjuj RestTemplate
    }

    // Metoda do pobierania obecnie zalogowanego użytkownika
    private User getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication != null && authentication.isAuthenticated() && !(authentication.getPrincipal() instanceof String)) {
            // Principal jest obiektem UserDetails. Jeśli nasza encja User implementuje UserDetails,
            // możemy bezpiecznie rzutować.
            Object principal = authentication.getPrincipal();
            if (principal instanceof UserDetails) { // Upewnij się, że to UserDetails (a nasza User implementuje)
                // Upewnij się, że obiekt principal jest faktycznie Twoją encją User
                // (To jest prawda, jeśli używasz UserDetailsServiceImpl zwracającego Twoją encję User)
                return (User) principal;
            }
        }
        return null; // Zwróć null, jeśli użytkownik nie jest zalogowany lub principal nie jest typu User
    }


    public BloodTestResult predictAndSave(BloodTestResult result) {
        // POBIERZ ZALOGOWANEGO UŻYTKOWNIKA I USTAWIJ GO W WYNIKU
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            // Obsłuż przypadek, gdy użytkownik nie jest zalogowany (np. rzuć wyjątek, zwróć null)
            // W naszej konfiguracji SecurityConfig ścieżka /anemia/form i /anemia/predict
            // wymagają uwierzytelnienia, więc to nie powinno się zdarzyć,
            // ale dobra praktyka to sprawdzić.
            throw new IllegalStateException("Użytkownik nie jest zalogowany, a próbuje wykonać predykcję!");
        }
        result.setUser(currentUser); // Ustaw użytkownika przed wysłaniem do FastAPI i zapisem

        // Wywołaj FastAPI
        PredictionResponse resp = callFastApi(result);
        result.setPrediction(resp.getPrediction());
        result.setProbability(resp.getProbability());
        // result.setEpicrisis(resp.getEpicrisis()); // Zakładając, że epicrisis jest w PredictionResponse

        // Zapisz wynik (teraz z ustawionym userem)
        return repository.save(result);
    }

    /**
     * Pobiera wyniki predykcji TYLKO dla zalogowanego użytkownika.
     */
    @Deprecated // Oznaczamy jako przestarzałe, bo nie używamy go już do historii
    public Iterable<BloodTestResult> getAll() {
        // Ta metoda już nie będzie używana do historii dla danego użytkownika
        return repository.findAll(); // Nadal działa, ale nie filtruje
    }

    // DODAJ TĘ NOWĄ METODĘ DO POBIERANIA HISTORII DLA ZALOGOWANEGO UŻYTKOWNIKA
    public Iterable<BloodTestResult> getHistoryForCurrentUser() {
        // POBIERZ ZALOGOWANEGO UŻYTKOWNIKA
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            // Jeśli użytkownik nie jest zalogowany (co nie powinno się zdarzyć dzięki SecurityConfig),
            // zwróć pustą listę lub rzuć wyjątek.
            return Collections.emptyList();
        }

        // Pobierz wyniki TYLKO dla tego użytkownika za pomocą nowej metody repozytorium
        return repository.findByUser(currentUser);
    }


    public PredictionResponse callFastApi(BloodTestResult input) {
        // RestTemplate restTemplate = new RestTemplate(); // Nie twórz nowego RestTemplate za każdym razem, użyj zainicjowanego w konstruktorze
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
        HttpEntity<Map<String, Double>> request = new HttpEntity<>(body, headers); // Użyj Map jako typu dla body

        ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                url,
                request, // Użyj obiektu HttpEntity
                PredictionResponse.class
        );

        return response.getBody();
    }


    /** Tylko zapis – bez wywoływania FastAPI */
    public BloodTestResult save(BloodTestResult result) {
        User currentUser = getCurrentUser();
        if (currentUser == null) {
            System.err.println("Warning: Saving BloodTestResult via /anemia/save without a logged-in user. This result will not be linked.");
            result.setUser(null); // Zapisz bez powiązania z userem, jeśli endpoint jest publiczny (nie jest w naszej obecnej konfig.)
        } else {
            result.setUser(currentUser); // Ustaw użytkownika
        }
        return repository.save(result);
    }


}