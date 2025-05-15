package com.agh.anemia.controller;

import com.agh.anemia.model.BloodTestResult;
import com.agh.anemia.service.AnemiaService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/anemia")
public class AnemiaController {
    @Autowired
    private AnemiaService anemiaService;

    @GetMapping("/form")
    public String showForm(Model model) {
        model.addAttribute("bloodTestResult", new BloodTestResult());
        return "form";
    }

    @PostMapping("/predict")       // scenariusz „Thymeleaf → FastAPI”
    public String predict(@ModelAttribute BloodTestResult bloodTestResult, Model model) {
        anemiaService.predictAndSave(bloodTestResult);
        model.addAttribute("result", bloodTestResult);
        return "result";
    }


    @GetMapping("/history")
    public String showHistory(Model model) {
        model.addAttribute("results", anemiaService.getAll());
        return "history";
    }

    @PostMapping("/save")          // wywoływane z fetch() w JS
    @ResponseBody
    public ResponseEntity<String> saveResult(@RequestBody BloodTestResult result) {
        System.out.println("Saving: " + result.toString()   );
        anemiaService.save(result);      // ← nie predictAndSave!
        return ResponseEntity.ok("Saved");
    }


}
