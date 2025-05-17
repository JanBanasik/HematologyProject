// src/main/resources/static/js/global.js

// FUNKCJA DO POBIERANIA TOKENU CSRF Z META TAGÓW (pozostała bez zmian)
function getCsrfToken() {
    const parameterNameMeta = document.querySelector('meta[name="_csrf_parameterName"]');
    const tokenMeta = document.querySelector('meta[name="_csrf_token"]');

    if (parameterNameMeta && tokenMeta) {
        return {
            parameterName: parameterNameMeta.content,
            token: tokenMeta.content
        };
    }
    return null;
}


document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle (pozostało bez zmian)
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const navbarMenu = document.getElementById('navbarMenu');

    if (hamburgerBtn && navbarMenu) {
        hamburgerBtn.addEventListener('click', function() {
            navbarMenu.classList.toggle('active'); // Używamy klasy 'active'
            const isExpanded = hamburgerBtn.getAttribute('aria-expanded') === 'true';
            hamburgerBtn.setAttribute('aria-expanded', !isExpanded);
            document.body.classList.toggle('menu-open');
            const navbarServicesLi = document.querySelector('.navbar-services');
            if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                navbarServicesLi.classList.remove('expanded');
            }
        });

        // Opcjonalnie: Zamknij menu po kliknięciu poza nim
        document.addEventListener('click', function(event) {
            const isClickInsideMenu = navbarMenu.contains(event.target) || hamburgerBtn.contains(event.target);
            const servicesDropdown = document.getElementById('servicesDropdown');
            const isClickInsideDropdown = servicesDropdown && servicesDropdown.contains(event.target);

            if (!isClickInsideMenu && !isClickInsideDropdown && navbarMenu.classList.contains('active')) {
                navbarMenu.classList.remove('active');
                hamburgerBtn.setAttribute('aria-expanded', 'false');
                document.body.classList.remove('menu-open');
                const navbarServicesLi = document.querySelector('.navbar-services');
                if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                    navbarServicesLi.classList.remove('expanded');
                }
            }
        });

        // Opcjonalnie: Zamknij główne menu po kliknięciu linku w menu
        navbarMenu.querySelectorAll('.navbar-menu a').forEach(link => {
            link.addEventListener('click', function() {
                setTimeout(() => {
                    navbarMenu.classList.remove('active');
                    hamburgerBtn.setAttribute('aria-expanded', 'false');
                    document.body.classList.remove('menu-open');
                    const navbarServicesLi = document.querySelector('.navbar-services');
                    if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                        navbarServicesLi.classList.remove('expanded');
                    }
                }, 100);
            });
        });
    }

    // Services Dropdown Toggle (pozostało bez zmian)
    const servicesBtn = document.getElementById('servicesBtn');
    const navbarServicesLi = servicesBtn ? servicesBtn.closest('.navbar-services') : null;

    if (servicesBtn && navbarServicesLi) {
        servicesBtn.addEventListener('click', function(e) {
            const navbarMenu = document.getElementById('navbarMenu');
            const isMobileView = window.getComputedStyle(hamburgerBtn).display !== 'none' || (navbarMenu && navbarMenu.classList.contains('active'));

            if (isMobileView) {
                e.preventDefault();
                navbarServicesLi.classList.toggle('expanded');
            }
        });

        // Opcjonalnie: Zamknij rozwijane menu Badań po kliknięciu w link wewnątrz niego (w trybie mobilnym)
        navbarServicesLi.querySelectorAll('.services-dropdown a').forEach(link => {
            link.addEventListener('click', function() {
                setTimeout(() => {
                    navbarServicesLi.classList.remove('expanded');
                    const navbarMenu = document.getElementById('navbarMenu');
                    const hamburgerBtn = document.getElementById('hamburgerBtn');
                    if (navbarMenu && navbarMenu.classList.contains('active')) {
                        navbarMenu.classList.remove('active');
                        if (hamburgerBtn) hamburgerBtn.setAttribute('aria-expanded', 'false');
                        document.body.classList.remove('menu-open');
                    }
                }, 100);
            });
        });
    }


    // Add smooth scrolling to all links (pozostało bez zmian)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Form field animations (pozostało bez zmian, dotyczy aktywnego stanu pola)
    const formGroups = document.querySelectorAll('.form-group');

    formGroups.forEach(group => {
        const input = group.querySelector('input, select, textarea');
        if (input) {
            input.addEventListener('focus', () => {
                group.classList.add('active');
            });

            input.addEventListener('blur', () => {
                if (!input.value) {
                    group.classList.remove('active');
                }
            });

            // Set active class on load if field has value
            if (input.value) {
                group.classList.add('active');
            }
        }
    });

    // Add page transition effect (dostosowane)
    const pageContent = document.querySelector('.page-content');
    if (pageContent) {
        // Usuń klasę, która może być dodana przez starsze skrypty
        // pageContent.classList.remove('fade-in'); // Jeśli używałeś innej nazwy

        // Dodaj klasę, która uruchomi animację z global.css
        pageContent.classList.add('animated');
        // CSS: .page-content { opacity: 0; transform: translateY(20px); } .page-content.animated { opacity: 1; transform: translateY(0); transition: opacity 0.5s ease, transform 0.5s ease; }
    }


    // Automatically hide alerts after 5 seconds (pozostało bez zmian)
    const alerts = document.querySelectorAll('.alert-danger, .alert-info');
    alerts.forEach(alert => {
        // Usunięto .animate-fade-in z html, style animacji są w global.css
        alert.classList.add('animated'); // Dodaj klasę animacji
        setTimeout(() => {
            // Zamiast bezpośredniego ustawiania opacity, dodaj klasę do ukrycia
            alert.classList.add('hide-animation');
            // W CSS: .alert-danger.hide-animation { opacity: 0; transition: opacity 0.5s ease-out; }
            // Opcjonalnie: Zastosuj display: none po zakończeniu transition
            setTimeout(() => {
                alert.style.display = 'none';
            }, 500); // Czas trwania animacji
        }, 5000);
    });


    // Add result animation (dostosowane)
    const resultElement = document.getElementById('result');
    if (resultElement && resultElement.innerHTML.trim() !== '') {
        // Usunięto .fade-in z html
        resultElement.classList.add('animated'); // Dodaj klasę animacji
        // CSS: #result { opacity: 0; transform: translateY(20px); } #result.animated { opacity: 1; transform: translateY(0); transition: opacity 0.5s ease, transform 0.5s ease; }
    }

    // Add active class to current nav link (pozostało bez zmian)
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-menu a');

    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath || (linkPath !== "/" && currentPath.startsWith(linkPath))) {
            link.classList.add('active');
        } else if (currentPath === "/" && linkPath === "/") {
            link.classList.add('active');
        }
    });


    // --- DODANA LOGIKA ANIMACJI FORMULARZY LOGOWANIA I REJESTRACJI ---
    // Wyszukaj wszystkie kontenery formularzy logowania i rejestracji
    const loginContainer = document.querySelector('.login-container');
    const registrationContainer = document.querySelector('.registration-container');

    if (loginContainer) {
        // Animuj kontener
        loginContainer.classList.add('animated');

        // Animuj grupy pól w formularzu
        const loginFormGroups = loginContainer.querySelectorAll('.login-form .form-group');
        loginFormGroups.forEach((group, index) => {
            // Dodaj klasę 'animated' z opóźnieniem
            setTimeout(() => {
                group.classList.add('animated');
            }, 100 + (index * 50)); // Opóźnienie 50ms między polami
        });

        // Animuj przycisk submit
        const loginButton = loginContainer.querySelector('.login-form button[type="submit"]');
        if (loginButton) {
            setTimeout(() => {
                loginButton.classList.add('animated');
            }, 100 + (loginFormGroups.length * 50) + 50); // Opóźnij po ostatnim polu
        }
    }

    if (registrationContainer) {
        // Animuj kontener
        registrationContainer.classList.add('animated');

        // Animuj grupy pól w formularzu
        const registrationFormGroups = registrationContainer.querySelectorAll('.registration-form .form-group');
        registrationFormGroups.forEach((group, index) => {
            // Dodaj klasę 'animated' z opóźnieniem
            setTimeout(() => {
                group.classList.add('animated');
            }, 100 + (index * 50)); // Opóźnienie 50ms między polami
        });

        // Animuj przycisk submit
        const registrationButton = registrationContainer.querySelector('.registration-form button[type="submit"]');
        if (registrationButton) {
            setTimeout(() => {
                registrationButton.classList.add('animated');
            }, 100 + (registrationFormGroups.length * 50) + 50); // Opóźnij po ostatnim polu
        }
    }

});

// --- KOD SKRYPTU FORMULARZA PREDYKCYJNEGO (TEN POWINIEN BYĆ W form.html) ---
/*
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('blood-form');

    if (form) {
        // ... (cała logika submit formularza, fetch do FastAPI i savePredictionResult) ...
        form.addEventListener('submit', async function (e) {
           e.preventDefault();
           // ... logika pobierania danych, wywołania FastAPI, pobierania tokena CSRF ...

           // Przy fetch do backendu (/anemia/savePredictionResult):
           // const csrf = typeof getCsrfToken === 'function' ? getCsrfToken() : null;
           // const headers = { 'Content-Type': 'application/json' };
           // if (csrf) { headers['X-XSRF-TOKEN'] = csrf.token; }
           // const saveResponse = await fetch('/anemia/savePredictionResult', { method: 'POST', headers: headers, body: JSON.stringify(resultData) });
           // ... reszta logiki obsługi odpowiedzi ...
        });
    }
});
*/