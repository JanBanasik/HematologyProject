// src/main/resources/static/js/global.js
document.addEventListener('DOMContentLoaded', function() {
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const navbarMenu = document.getElementById('navbarMenu');
    const servicesBtn = document.getElementById('servicesBtn');
    const navbarServicesLi = servicesBtn ? servicesBtn.closest('.navbar-services') : null;

    // Obsługa kliknięcia hamburgera
    if (hamburgerBtn && navbarMenu) {
        hamburgerBtn.addEventListener('click', function() {
            // Toggle klasy 'active' na liście menu (dla panelu mobilnego)
            navbarMenu.classList.toggle('active');

            // Zmień atrybuty ARIA
            const isExpanded = hamburgerBtn.getAttribute('aria-expanded') === 'true';
            hamburgerBtn.setAttribute('aria-expanded', !isExpanded);

            // Zablokuj scrollowanie tła
            document.body.classList.toggle('menu-open');

            // Zwiń menu Badań po otwarciu/zamknięciu głównego menu mobilnego
            if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                navbarServicesLi.classList.remove('expanded');
            }
        });

        // Opcjonalnie: Zamknij menu po kliknięciu poza nim
        document.addEventListener('click', function(event) {
            const isClickInsideMenu = navbarMenu.contains(event.target) || hamburgerBtn.contains(event.target);

            if (!isClickInsideMenu && navbarMenu.classList.contains('active')) {
                navbarMenu.classList.remove('active');
                hamburgerBtn.setAttribute('aria-expanded', 'false');
                document.body.classList.remove('menu-open');
            }
        });

        // Opcjonalnie: Zamknij główne menu po kliknięciu linku
        navbarMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', function() {
                // Opóźnij zamknięcie menu, aby link miał czas zadziałać
                setTimeout(() => {
                    navbarMenu.classList.remove('active');
                    hamburgerBtn.setAttribute('aria-expanded', 'false');
                    document.body.classList.remove('menu-open');
                    // Zwiń menu Badań po zamknięciu głównego menu
                    if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                        navbarServicesLi.classList.remove('expanded');
                    }
                }, 100); // Krótkie opóźnienie
            });
        });
    }

    // Logika rozwijania/zwijania menu Badań w trybie mobilnym (po kliknięciu przycisku "Badania")
    if (servicesBtn && navbarServicesLi) {
        servicesBtn.addEventListener('click', function(event) {
            // Zapobiegaj domyślnej akcji (np. nawigacji, jeśli przycisk byłby linkiem)
            event.preventDefault();
            // Toggle klasy 'expanded' na nadrzędnym li
            navbarServicesLi.classList.toggle('expanded');
        });
    }

    // Logika JS do pobierania ciasteczka CSRF i dodawania go do nagłówków fetch
    // Ta funkcja jest potrzebna w form.html, jeśli używasz tam fetch do /anemia/save
    // Możesz ją przenieść do tego pliku global.js i wywołać z form.html
    // lub zostawić w form.html, jeśli tylko tam jest używana.
    // Poniżej kod tylko jako przypomnienie:
    /*
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }
    // W form.html, przy fetch do /anemia/save:
    // const csrfToken = getCookie('XSRF-TOKEN');
    // const headers = { 'Content-Type': 'application/json', 'X-XSRF-TOKEN': csrfToken };
    // fetch('/anemia/save', { method: 'POST', headers: headers, body: JSON.stringify(resultData) });
    */
});