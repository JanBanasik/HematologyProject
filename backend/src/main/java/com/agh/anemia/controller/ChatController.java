package com.agh.anemia.controller;
import com.agh.anemia.dto.FastApiChatRequest;
import com.agh.anemia.model.ChatMessageRequest;
import com.agh.anemia.model.ChatMessageResponse;
import com.agh.anemia.model.FastApiChatResponse;
import com.agh.anemia.service.UserDetailsServiceImpl;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;



@RestController
@RequestMapping("/chat")
@PreAuthorize("isAuthenticated()")
public class ChatController {

    // Zamiast WebClient, używamy RestTemplate
    private final RestTemplate restTemplate; // Zadeklaruj RestTemplate

    @Value("${fastapi.chat.url}")
    private String fastapiBaseUrl; // Zmień nazwę zmiennej na baseUrl, bo RestTemplate.exchange przyjmuje pełny URI

    // Jeśli potrzebujesz UserDetailsServiceImpl w konstruktorze (jak w Twoim oryginalnym kodzie)
    private final UserDetailsServiceImpl userDetailsServiceImpl;


    public ChatController(RestTemplate restTemplate, UserDetailsServiceImpl userDetailsServiceImpl) { // Wstrzyknij RestTemplate (możesz go zdefiniować jako Bean)
        this.restTemplate = restTemplate;
        this.userDetailsServiceImpl = userDetailsServiceImpl; // Przyjmij, że to jest potrzebne z Twojego oryginalnego konstruktora
    }





    @PostMapping("/send")
    public ResponseEntity<ChatMessageResponse> handleChatMessage(@RequestBody ChatMessageRequest request) {
        String userMessage = request.getMessage();
        System.out.println("Received chat message from frontend: " + userMessage);

        // Utwórz obiekt żądania dla FastAPI
        FastApiChatRequest fastapiRequest = new FastApiChatRequest();
        fastapiRequest.setMessage(userMessage);

        // Ustaw nagłówki dla żądania do FastAPI (np. Content-Type)
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Utwórz HttpEntity zawierający ciało i nagłówki
        HttpEntity<FastApiChatRequest> httpEntity = new HttpEntity<>(fastapiRequest, headers);

        String fastapiEndpointUrl = fastapiBaseUrl + "/generate"; // Pełny URL endpointu FastAPI

        // --- Wysyłanie żądania do FastAPI za pomocą RestTemplate ---
        FastApiChatResponse fastapiResponse = null;
        try {
            // Wysłanie żądania POST i oczekiwanie na odpowiedź
            ResponseEntity<FastApiChatResponse> response = restTemplate.postForEntity(
                    fastapiEndpointUrl, // URL endpointu FastAPI
                    httpEntity,         // Ciało żądania z nagłówkami
                    FastApiChatResponse.class // Oczekiwany typ odpowiedzi
            );

            if (response.getStatusCode().is2xxSuccessful()) {
                fastapiResponse = response.getBody();
            } else {
                // Obsługa błędów HTTP zwróconych przez FastAPI (np. 4xx, 5xx)
                System.err.println("FastAPI returned non-successful status: " + response.getStatusCode());
                // Możesz próbować odczytać ciało błędu, jeśli FastAPI je zwraca
                // String errorBody = restTemplate.exchange(...) lub pobrać go z response.getBody() jeśli FastApiChatResponse ma pole błędu
                throw new RuntimeException("FastAPI error response: " + response.getStatusCode());
            }

        } catch (HttpClientErrorException e) {
            // Obsługa błędów klienta (4xx) z FastAPI
            System.err.println("HTTP Client Error from FastAPI: " + e.getStatusCode() + " - " + e.getResponseBodyAsString());
            return ResponseEntity.status(e.getStatusCode()).body(new ChatMessageResponse("FastAPI error: " + e.getResponseBodyAsString()));
        } catch (HttpServerErrorException e) {
            // Obsługa błędów serwera (5xx) z FastAPI
            System.err.println("HTTP Server Error from FastAPI: " + e.getStatusCode() + " - " + e.getResponseBodyAsString());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(new ChatMessageResponse("FastAPI server error: " + e.getResponseBodyAsString()));
        } catch (ResourceAccessException e) {
            // Obsługa błędów sieciowych (np. FastAPI jest wyłączone, brak połączenia)
            System.err.println("Network or Resource Access Error communicating with FastAPI: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(new ChatMessageResponse("Błąd połączenia z serwisem AI."));
        } catch (Exception e) {
            // Obsługa innych nieoczekiwanych błędów
            System.err.println("Unexpected error during FastAPI communication: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(new ChatMessageResponse("Wystąpił błąd podczas komunikacji z asystentem AI."));
        }


        // Przetwórz odpowiedź z FastAPI i utwórz odpowiedź dla frontendu
        if (fastapiResponse != null && fastapiResponse.getReply() != null) {
            System.out.println("Received response from FastAPI: " + fastapiResponse.getReply());
            ChatMessageResponse frontendResponse = new ChatMessageResponse(fastapiResponse.getReply());
            return ResponseEntity.ok(frontendResponse); // Zwróć odpowiedź dla frontendu
        } else {
            System.err.println("Invalid or empty response from FastAPI.");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(new ChatMessageResponse("Otrzymano nieprawidłową odpowiedź od serwisu AI."));
        }
    }

    // Jeśli używasz autowired, nie potrzebujesz wstrzykiwać w konstruktorze
    // @Autowired
    // private RestTemplate restTemplate;

}