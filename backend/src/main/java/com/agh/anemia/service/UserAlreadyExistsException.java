package com.agh.anemia.service;

public class UserAlreadyExistsException extends RuntimeException {
    public UserAlreadyExistsException(String message) {
        super(message);
    }
    // Można dodać konstruktor przyjmujący username i email, aby zbudować bardziej specyficzny komunikat
    public UserAlreadyExistsException(String username, String email, boolean usernameExists, boolean emailExists) {
        super(buildMessage(username, email, usernameExists, emailExists));
    }

    private static String buildMessage(String username, String email, boolean usernameExists, boolean emailExists) {
        if (usernameExists && emailExists) {
            return "Użytkownik o nazwie '" + username + "' oraz adres email '" + email + "' już istnieją.";
        } else if (usernameExists) {
            return "Użytkownik o nazwie '" + username + "' już istnieje.";
        } else if (emailExists) {
            return "Użytkownik o adresie email '" + email + "' już istnieje.";
        } else {
            return "Użytkownik już istnieje (nieznany powód)."; // Komunikat awaryjny
        }
    }
}