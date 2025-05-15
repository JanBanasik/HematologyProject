// src/main/java/com/agh/anemia/service/UserDetailsServiceImpl.java (lub security)
package com.agh.anemia.service; // Lub com.agh.anemia.security

import com.agh.anemia.model.User;
import com.agh.anemia.repository.UserRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service // Oznacz jako komponent Spring
public class UserDetailsServiceImpl implements UserDetailsService {

    private final UserRepository userRepository;

    // Wstrzyknij UserRepository przez konstruktor
    public UserDetailsServiceImpl(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // Znajdź użytkownika w bazie danych po nazwie użytkownika
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("Użytkownik nie znaleziony: " + username));

        // Spring Security wymaga obiektu UserDetails.
        // Ponieważ nasza encja User implementuje UserDetails, możemy ją zwrócić bezpośrednio.
        return user;
    }
}