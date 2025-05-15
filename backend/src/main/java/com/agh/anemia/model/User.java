// src/main/java/com/agh/anemia/model/User.java
package com.agh.anemia.model;

import jakarta.persistence.*;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
import java.util.Collections; // Dla prostego przypadku jednej roli

@Entity
@Table(name = "users") // Domyślnie nazwa tabeli byłaby "user", co może być słowem kluczowym SQL
public class User implements UserDetails { // Implementujemy UserDetails dla Spring Security
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true) // Nazwa użytkownika musi być unikalna
    private String username;

    @Column(nullable = false)
    private String password; // Zapisane hasło będzie zaszyfrowane (haszowane)

    // Można dodać inne pola, np. email, imię, nazwisko

    // Dla uproszczenia, przechowujemy rolę jako String.
    // W bardziej złożonej aplikacji można użyć encji Role i relacji ManyToMany.
    private String role = "USER"; // Domyślna rola to USER

    // Konstruktor domyślny wymagany przez JPA
    public User() {
    }

    public User(String username, String password, String role) {
        this.username = username;
        this.password = password;
        this.role = role;
    }

    // Getters and Setters

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    @Override
    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    @Override
    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }


    // Implementacja metod UserDetails dla Spring Security

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        // Zwraca kolekcję uprawnień (ról) użytkownika
        return Collections.singletonList(new SimpleGrantedAuthority("ROLE_" + role)); // Rola w Spring Security musi mieć prefiks "ROLE_"
    }

    @Override
    public boolean isAccountNonExpired() {
        return true; // Konto nigdy nie wygasa w tym przykładzie
    }

    @Override
    public boolean isAccountNonLocked() {
        return true; // Konto nigdy nie jest blokowane w tym przykładzie
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true; // Poświadczenia (hasło) nigdy nie wygasają w tym przykładzie
    }

    @Override
    public boolean isEnabled() {
        return true; // Konto jest zawsze aktywne w tym przykładzie
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", role='" + role + '\'' +
                '}';
    }
}