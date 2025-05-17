// src/main/java/com/agh/anemia/service/RegistrationService.java
package com.agh.anemia.service;

import com.agh.anemia.model.User;
import com.agh.anemia.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;


@Service
@Transactional // Metody serwisu modyfikujące dane powinny być transakcyjne
public class RegistrationService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder; // Wstrzykujemy koder haseł

    // Wstrzyknij zależności przez konstruktor (zalecane)
    public RegistrationService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Rejestruje nowego użytkownika po sprawdzeniu unikalności nazwy użytkownika i emaila.
     * @param user DANE użytkownika (username, email, RAW password)
     * @return Zapisany obiekt User
     * @throws UserAlreadyExistsException jeśli użytkownik o podanej nazwie lub emailu już istnieje
     */
    @Transactional // Jawnie oznacz metodę jako transakcyjną, nawet jeśli klasa ma @Transactional
    public User registerNewUser(User user) {
        // Sprawdź, czy użytkownik o podanej nazwie użytkownika lub emailu już istnieje
        boolean usernameExists = userRepository.existsByUsername(user.getUsername());
        boolean emailExists = userRepository.existsByEmail(user.getEmail());

        if (usernameExists || emailExists) {
            // Użyj zaktualizowanego konstruktora wyjątku, aby podać szczegółowy komunikat
            throw new UserAlreadyExistsException(user.getUsername(), user.getEmail(), usernameExists, emailExists);
        }

        // Zaszyfruj hasło przed zapisaniem do bazy
        user.setPassword(passwordEncoder.encode(user.getPassword()));

        // Ustaw domyślną rolę (jeśli nie została ustawiona wcześniej w obiekcie User)
        if (user.getRole() == null || user.getRole().isEmpty()) {
            user.setRole("USER");
        }

        // Zapisz użytkownika do bazy danych
        User savedUser = userRepository.save(user);
        System.out.println("User registered successfully: " + savedUser.getUsername());
        return savedUser;
    }

    // Możesz dodać inne metody serwisu, jeśli potrzebujesz, np. do pobierania usera
    public Optional<User> findByUsername(String username) {
        return userRepository.findByUsername(username);
    }
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }

}