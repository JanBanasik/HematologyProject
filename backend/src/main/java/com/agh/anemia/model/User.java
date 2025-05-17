// src/main/java/com/agh/anemia/model/User.java
package com.agh.anemia.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
import java.util.Collections;

@Getter
@Setter
@Entity
@Table(name = "users") // Domyślnie nazwa tabeli byłaby "user", co może być słowem kluczowym SQL
public class User implements UserDetails {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true) // Nazwa użytkownika musi być unikalna
    private String username;

    @Column(nullable = false)
    private String password; // Zapisane hasło będzie zaszyfrowane (haszowane)

    // GETTERY I SETTERY DLA NOWYCH PÓL
    // DODAJ LUB ZMODYFIKUJ POLE EMAIL Z OGRANICZENIEM UNIKALNOŚCI

    @Column(nullable = false, unique = true) // Email musi być unikalny i nie może być NULL
    private String email;

    // Możesz dodać inne pola, np. imię, nazwisko (widzę je w Twoim register.html)

    private String firstName;

    private String lastName;

    private String role = "USER"; // Domyślna rola to USER

    // Konstruktor domyślny wymagany przez JPA
    public User() {
    }

    // Zaktualizuj konstruktory, jeśli dodajesz nowe pola
    public User(String username, String password, String role, String email, String firstName, String lastName) {
        this.username = username;
        this.password = password;
        this.role = role;
        this.email = email;
        this.firstName = firstName;
        this.lastName = lastName;
    }


    // Getters and Setters (dla wszystkich pól, w tym nowych)

    @Override
    public String getUsername() { return username; }

    @Override
    public String getPassword() { return password; }


    // Implementacja metod UserDetails dla Spring Security (bez zmian)
    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return Collections.singletonList(new SimpleGrantedAuthority("ROLE_" + role));
    }

    @Override public boolean isAccountNonExpired() { return true; }
    @Override public boolean isAccountNonLocked() { return true; }
    @Override public boolean isCredentialsNonExpired() { return true; }
    @Override public boolean isEnabled() { return true; }


    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", email='" + email + '\'' + // Dodaj email do toString
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", role='" + role + '\'' +
                '}';
    }
}