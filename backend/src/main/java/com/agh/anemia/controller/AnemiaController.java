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
import org.springframework.web.server.ResponseStatusException;

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

        BloodTestResult savedResult = anemiaService.predictAndSave(bloodTestResult);
        model.addAttribute("result", savedResult);
        return "result";
    }


    @GetMapping("/history")
    public String showHistory(Model model) {
        model.addAttribute("results", anemiaService.getHistoryForCurrentUser());
        return "history";
    }

    @GetMapping("/history/details/{id}")
    @ResponseBody
    public BloodTestResult getResultDetails(@PathVariable Long id) {
        BloodTestResult result = anemiaService.getResultByIdAndUser(id);
        if (result == null) {
            throw new ResponseStatusException(
                    HttpStatus.NOT_FOUND, "Result Not Found or access denied");
        }
        return result;
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