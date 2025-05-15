package com.agh.anemia.service;

// Klasa wyjątku dla istniejącego użytkownika (opcjonalne, ale dobra praktyka)
public class UserAlreadyExistsException extends RuntimeException {
    public UserAlreadyExistsException(String username) {
        super("Użytkownik o nazwie '" + username + "' już istnieje.");
    }
}
