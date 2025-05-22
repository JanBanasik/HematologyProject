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
    private final PasswordEncoder passwordEncoder;

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
    @Transactional
    public User registerNewUser(User user) {

        boolean usernameExists = userRepository.existsByUsername(user.getUsername());
        boolean emailExists = userRepository.existsByEmail(user.getEmail());

        if (usernameExists || emailExists) {
            throw new UserAlreadyExistsException(user.getUsername(), user.getEmail(), usernameExists, emailExists);
        }

        user.setPassword(passwordEncoder.encode(user.getPassword()));

        if (user.getRole() == null || user.getRole().isEmpty()) {
            user.setRole("USER");
        }

        User savedUser = userRepository.save(user);
        System.out.println("User registered successfully: " + savedUser.getUsername());
        return savedUser;
    }

    public Optional<User> findByUsername(String username) {
        return userRepository.findByUsername(username);
    }
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }

}