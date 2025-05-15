// src/main/java/com/agh/anemia/config/SecurityConfig.java
package com.agh.anemia;

import com.agh.anemia.service.UserDetailsServiceImpl; // Importuj swoją implementację
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

// Dodaj importy dla CookieCsrfTokenRepository i CsrfTokenRepository
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;
import org.springframework.security.web.csrf.CsrfTokenRepository;

import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final UserDetailsServiceImpl userDetailsServiceImpl;

    public SecurityConfig(UserDetailsServiceImpl userDetailsServiceImpl) {
        this.userDetailsServiceImpl = userDetailsServiceImpl;
    }

    /**
     * Konfiguruje repozytorium tokenów CSRF do używania ciasteczek.
     * withHttpOnlyFalse() pozwala JavaScriptowi odczytać ciasteczko XSRF-TOKEN,
     * co jest standardem przy AJAX/fetch.
     */
    @Bean
    public CsrfTokenRepository csrfTokenRepository() {
        // Domyślna nazwa ciasteczka to "XSRF-TOKEN"
        CookieCsrfTokenRepository repository = CookieCsrfTokenRepository.withHttpOnlyFalse();
        // Opcjonalnie: ustaw ścieżkę ciasteczka, np. na "/", jeśli ma być dostępne wszędzie
        repository.setCookiePath("/"); // Ustawienie ścieżki ciasteczka na root
        return repository;
    }


    /**
     * Konfiguruje łańcuch filtrów bezpieczeństwa HTTP.
     * Definiuje reguły autoryzacji dla różnych ścieżek URL.
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                // Konfiguruj CSRF do używania naszego CookieCsrfTokenRepository
                // TA LINIA ZOSTAŁA ZMODYFIKOWANA
                .csrf(csrf -> csrf.csrfTokenRepository(csrfTokenRepository()))

                .authorizeHttpRequests(authorize -> authorize
                        // Dozwól dostęp publiczny do strony głównej, strony logowania, strony rejestracji i zasobów statycznych
                        .requestMatchers("/", "/login", "/register", "/css/**", "/js/**").permitAll()
                        // Wszelkie inne żądania wymagają uwierzytelnienia
                        .anyRequest().authenticated()
                )
                .formLogin(form -> form
                        .loginPage("/login")
                        // Po udanym zalogowaniu przekieruj na adres, który użytkownik próbował odwiedzić,
                        // lub na stronę główną ("/") jeśli nie było docelowego adresu.
                        .defaultSuccessUrl("/", true)
                        .permitAll()
                )
                .logout(logout -> logout
                        .logoutSuccessUrl("/")
                        .permitAll()
                );

        return http.build();
    }

    /**
     * Definiuje AuthenticationProvider używający naszego UserDetailsService i PasswordEncoder.
     * Spring Security automatycznie użyje tego Providera do uwierzytelniania.
     */
    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        // Ustaw naszą implementację UserDetailsService
        authProvider.setUserDetailsService(userDetailsServiceImpl);
        // Ustaw nasz PasswordEncoder
        authProvider.setPasswordEncoder(passwordEncoder());
        return authProvider;
    }


    /**
     * Konfiguruje koder haseł.
     * BCryptPasswordEncoder jest zalecany do bezpiecznego przechowywania haseł.
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    // Usunięto bean userDetailsService z InMemoryUserDetailsManager
}