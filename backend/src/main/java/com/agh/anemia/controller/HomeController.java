// src/main/java/com/agh/anemia/controller/HomeController.java
package com.agh.anemia.controller;

import com.agh.anemia.model.User;
import com.agh.anemia.service.RegistrationService;
import com.agh.anemia.service.UserAlreadyExistsException; // Importuj wyjątek
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.servlet.mvc.support.RedirectAttributes; // Do przekazywania komunikatów po przekierowaniu


@Controller
public class HomeController {

    private final RegistrationService registrationService;

    // Wstrzyknij RegistrationService
    public HomeController(RegistrationService registrationService) {
        this.registrationService = registrationService;
    }

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @GetMapping("/login")
    public String login() {
        return "login";
    }

    /**
     * Wyświetla formularz rejestracji.
     */
    @GetMapping("/register")
    public String showRegistrationForm(Model model) {
        model.addAttribute("user", new User()); // Dodaj pusty obiekt User do formularza
        return "register";
    }

    /**
     * Obsługuje przesłanie formularza rejestracji.
     */
    @PostMapping("/register")
    public String registerUser(@ModelAttribute("user") User user,
                               Model model, // Używamy Model, żeby dodać atrybut błędu, jeśli rejestracja się nie uda (zostajemy na tej samej stronie)
                               RedirectAttributes redirectAttributes) { // Używamy RedirectAttributes, żeby dodać komunikat po udanej rejestracji (przekierowanie na login)

        try {
            registrationService.registerNewUser(user);
            // Po udanej rejestracji przekieruj na stronę logowania z komunikatem sukcesu
            redirectAttributes.addFlashAttribute("registrationSuccess", true);
            redirectAttributes.addFlashAttribute("username", user.getUsername());
            return "redirect:/login";
        } catch (UserAlreadyExistsException e) {
            // Jeśli użytkownik istnieje, dodaj błąd do modelu i wróć do formularza rejestracji
            model.addAttribute("registrationError", e.getMessage());
            // Zwracamy "register", aby pozostać na tej samej stronie i wyświetlić błąd
            return "register";
        } catch (Exception e) {
            // Obsługa innych potencjalnych błędów
            model.addAttribute("registrationError", "Wystąpił błąd podczas rejestracji: " + e.getMessage());
            return "register";
        }
    }
}