// src/main/java/com/agh/anemia/controller/AnemiaController.java
package com.agh.anemia.controller;

import com.agh.anemia.dto.BloodTestPredictionDto;
import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.service.AnemiaService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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

    @PostMapping("/savePredictionResult") // Zmieniono z "/save"
    @ResponseBody
    public ResponseEntity<String> saveResult(@RequestBody BloodTestPredictionDto resultDto) {
        System.out.println("Received BloodTestPredictionDto in controller for saving: " + resultDto.toString());

        try {
            anemiaService.savePredictionResult(resultDto); // Wywołaj metodę serwisu, która przyjmuje DTO

            System.out.println("BloodTestPredictionDto processed and saved successfully.");
            return ResponseEntity.ok("Saved");
        } catch (Exception e) {
            System.err.println("Error processing BloodTestPredictionDto: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Error saving result: " + e.getMessage());
        }
    }


}