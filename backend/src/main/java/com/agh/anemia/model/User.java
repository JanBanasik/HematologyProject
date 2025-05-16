// src/main/java/com/agh/anemia/model/User.java
package com.agh.anemia.model;

import jakarta.persistence.*;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
import java.util.Collections; // Dla prostego przypadku jednej roli

@Entity
@Table(name = "users") // Nazwa tabeli w bazie danych
public class User implements UserDetails {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String username;

    @Column(nullable = false)
    private String password; // Zapisane hasło będzie zaszyfrowane

    // DODAJ NOWE POLA
    private String firstName;
    private String lastName;
    private String email; // Można dodać validację formatu emaila, ale na razie pomińmy dla prostoty


    private String role = "USER"; // Domyślna rola

    // Relacja One-to-Many do wyników badań (opcjonalne do zdefiniowania w encji User,
    // ale jasniej pokazuje model - jeden user ma wiele wyników)
    // @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    // private List<BloodTestResult> bloodTestResults; // Zostawiamy zakomentowane dla prostoty


    // Konstruktor domyślny wymagany przez JPA
    public User() {
    }

    // Konstruktor dla tworzenia użytkownika (można dodać więcej pól)
    public User(String username, String password) {
        this.username = username;
        this.password = password;
        this.role = "USER";
    }
    // Konstruktor z wszystkimi polami (opcjonalny)
    public User(String username, String password, String firstName, String lastName, String email) {
        this.username = username;
        this.password = password;
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.role = "USER";
    }


    // Getters and Setters (dla wszystkich pól)

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    @Override public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    @Override public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
    public String getFirstName() { return firstName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }
    public String getLastName() { return lastName; }
    public void setLastName(String lastName) { this.lastName = lastName; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    // Getters dla UserDetails
    @Override public Collection<? extends GrantedAuthority> getAuthorities() { return Collections.singletonList(new SimpleGrantedAuthority("ROLE_" + role)); }
    @Override public boolean isAccountNonExpired() { return true; }
    @Override public boolean isAccountNonLocked() { return true; }
    @Override public boolean isCredentialsNonExpired() { return true; }
    @Override public boolean isEnabled() { return true; }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", email='" + email + '\'' +
                ", role='" + role + '\'' +
                '}';
    }
}