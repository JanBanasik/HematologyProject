// src/main/java/com/agh/anemia/service/RegistrationService.java
package com.agh.anemia.service;

import com.agh.anemia.model.User;
import com.agh.anemia.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;


@Service
public class RegistrationService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder; // Wstrzykujemy koder haseł

    public RegistrationService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Rejestruje nowego użytkownika.
     * @param user DANE użytkownika (username, RAW password)
     * @return Zapisany obiekt User
     * @throws UserAlreadyExistsException jeśli użytkownik o podanej nazwie już istnieje
     */
    public User registerNewUser(User user) {
        // Sprawdź, czy użytkownik o podanej nazwie już istnieje
        if (userRepository.existsByUsername(user.getUsername())) {
            throw new UserAlreadyExistsException(user.getUsername());
        }

        // Zaszyfruj hasło przed zapisaniem do bazy
        user.setPassword(passwordEncoder.encode(user.getPassword()));

        // Ustaw domyślną rolę (jeśli nie została ustawiona wcześniej w obiekcie User)
        if (user.getRole() == null || user.getRole().isEmpty()) {
            user.setRole("USER");
        }

        // Zapisz użytkownika do bazy danych
        return userRepository.save(user);
    }
}