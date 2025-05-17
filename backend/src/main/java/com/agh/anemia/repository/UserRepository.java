// src/main/java/com/agh/anemia/repository/UserRepository.java
package com.agh.anemia.repository;

import com.agh.anemia.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

// Spring Data JPA automatycznie utworzy implementację tego interfejsu
public interface UserRepository extends JpaRepository<User, Long> {
    // Metoda do znajdowania użytkownika po nazwie użytkownika.
    // Spring Data JPA automatycznie generuje zapytanie SQL na podstawie nazwy metody.
    Optional<User> findByUsername(String username);

    // Opcjonalnie, metoda do sprawdzenia, czy użytkownik o danej nazwie istnieje
    boolean existsByUsername(String username);

    boolean existsByEmail(String email);
}