// src/main/java/com/agh/anemia/controller/AnemiaController.java
package com.agh.anemia.controller;

import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.model.User;
import com.agh.anemia.service.AnemiaService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/anemia")
public class AnemiaController {

    // Wstrzyknij AnemiaService przez konstruktor (zalecane zamiast @Autowired na polu)
    private final AnemiaService anemiaService;

    public AnemiaController(AnemiaService anemiaService) {
        this.anemiaService = anemiaService;
    }


    @GetMapping("/form")
    public String showForm(Model model) {
        model.addAttribute("bloodTestResult", new BloodTestResult());
        return "form";
    }

    @PostMapping("/predict") // Scenariusz Thymeleaf -> FastAPI
    public String predict(@ModelAttribute BloodTestResult bloodTestResult, Model model) {
        // Metoda predictAndSave w serwisie sama pobierze zalogowanego usera
        BloodTestResult savedResult = anemiaService.predictAndSave(bloodTestResult);
        model.addAttribute("result", savedResult);
        // Możesz przekierować na stronę result lub pozostać na form i wyświetlić wynik
        // Jeśli chcesz przekierować na dedykowaną stronę wyniku:
        // return "redirect:/anemia/result/" + savedResult.getId(); // Wymagałoby nowego endpointu
        // Jeśli zostajesz na tej samej stronie (form.html z JS):
        return "result"; // Domyślne przejście do result.html - to działa z aktualnym setupem
    }


    @GetMapping("/history")
    public String showHistory(Model model) {
        // ZMIENIONO WYWOŁANIE SERWISU: TERAZ POBIERA TYLKO HISTORIĘ DLA ZALOGOWANEGO UŻYTKOWNIKA
        model.addAttribute("results", anemiaService.getHistoryForCurrentUser());
        return "history";
    }

    @PostMapping("/save") // Wywoływane z fetch() w JS w form.html
    @ResponseBody // Odpowiada bezpośrednio danymi, a nie nazwą widoku
    public BloodTestResult save(BloodTestResult result) {
        User currentUser = getCurrentUser(); // Pobierz usera

// *** TE LOGI POWINNY SIĘ POJAWIĆ W KONSOLI SERWERA ***
        if (currentUser != null) {
            System.out.println("Attempting to save BloodTestResult for user: " + currentUser.getUsername());
        } else {
            System.err.println("Attempting to save BloodTestResult for NULL user!"); // Czy to się pojawia? Nie powinno jeśli zalogowany
        }
        System.out.println("BloodTestResult data BEFORE setting user: " + result.toString());

        if (currentUser == null) {
            // ... obsługa null ...
            result.setUser(null);
        } else {
            result.setUser(currentUser); // Ustaw użytkownika
        }

        System.out.println("BloodTestResult data AFTER setting user: " + result.toString()); // Czy user jest ustawiony w toString?

        System.out.println("Calling repository.save..."); // *** TEN LOG MUSI SIĘ POJAWIĆ ***
        BloodTestResult savedResult = anemiaService.save(result); // Zapis do bazy
        System.out.println("Repository.save called. Saved result ID: " + savedResult.getId()); // *** TEN LOG MUSI SIĘ POJAWIĆ JEŚLI ZAPIS SIĘ UDAŁ ***

        return savedResult; // Zwróć zapisany obiekt
    }


    private User getCurrentUser() {
        // Pobierz obiekt Authentication z kontekstu bezpieczeństwa Spring Security
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        // Sprawdź, czy uwierzytelnienie istnieje i czy użytkownik jest uwierzytelniony (nie anonimowy domyślnie)
        // i czy principal nie jest po prostu Stringiem (jak "anonymousUser")
        if (authentication != null && authentication.isAuthenticated() && !(authentication.getPrincipal() instanceof String)) {
            // Principal jest obiektem UserDetails. Ponieważ nasza encja User implementuje UserDetails,
            // możemy bezpiecznie rzutować na User.
            Object principal = authentication.getPrincipal();
            if (principal instanceof UserDetails) {
                // W naszym przypadku, UserDetailsServiceImpl zwraca naszą encję User
                return (User) principal;
            }
        }
        // Zwróć null, jeśli użytkownik nie jest zalogowany lub principal nie jest oczekiwanego typu UserDetails
        return null;
    }

}