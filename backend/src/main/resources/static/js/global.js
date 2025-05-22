function getCsrfToken() {
    const parameterNameMeta = document.querySelector('meta[name="_csrf_parameterName"]');
    const tokenMeta = document.querySelector('meta[name="_csrf_token"]');

    const headerNameMeta = document.querySelector('meta[name="_csrf_headerName"]');

    if (parameterNameMeta && tokenMeta && headerNameMeta) {

        return {
            parameterName: parameterNameMeta.content,
            token: tokenMeta.content,
            headerName: headerNameMeta.content
        };
    }
    console.warn("CSRF meta tags not fully found. Cannot build CSRF token object.");
    return null;
}


document.addEventListener('DOMContentLoaded', function() {

    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const navbarMenu = document.getElementById('navbarMenu');

    if (hamburgerBtn && navbarMenu) {
        hamburgerBtn.addEventListener('click', function() {
            navbarMenu.classList.toggle('active');
            const isExpanded = hamburgerBtn.getAttribute('aria-expanded') === 'true';
            hamburgerBtn.setAttribute('aria-expanded', !isExpanded);
            document.body.classList.toggle('menu-open');
            const navbarServicesLi = document.querySelector('.navbar-services');
            if (navbarServicesLi && navbarServicesLi.classList.contains('expanded')) {
                navbarServicesLi.classList.remove('expanded');
            }
        });

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

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

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

    const pageContent = document.querySelector('.page-content');
    if (pageContent) {
        pageContent.classList.add('animated');
    }

    const alerts = document.querySelectorAll('.alert-danger, .alert-info');
    alerts.forEach(alert => {

        alert.classList.add('animated');
        setTimeout(() => {

            alert.classList.add('hide-animation');

            setTimeout(() => {
                alert.style.display = 'none';
            }, 500);
        }, 5000);
    });

    const resultElement = document.getElementById('result');
    if (resultElement && resultElement.innerHTML.trim() !== '') {

        resultElement.classList.add('animated'); // Dodaj klasę animacji
    }

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

    const loginContainer = document.querySelector('.login-container');
    const registrationContainer = document.querySelector('.registration-container');

    if (loginContainer) {

        loginContainer.classList.add('animated');

        const loginFormGroups = loginContainer.querySelectorAll('.login-form .form-group');
        loginFormGroups.forEach((group, index) => {
            // Dodaj klasę 'animated' z opóźnieniem
            setTimeout(() => {
                group.classList.add('animated');
            }, 100 + (index * 50));
        });

        const loginButton = loginContainer.querySelector('.login-form button[type="submit"]');
        if (loginButton) {
            setTimeout(() => {
                loginButton.classList.add('animated');
            }, 100 + (loginFormGroups.length * 50) + 50);
        }
    }

    if (registrationContainer) {
        registrationContainer.classList.add('animated');

        const registrationFormGroups = registrationContainer.querySelectorAll('.registration-form .form-group');
        registrationFormGroups.forEach((group, index) => {

            setTimeout(() => {
                group.classList.add('animated');
            }, 100 + (index * 50));
        });

        const registrationButton = registrationContainer.querySelector('.registration-form button[type="submit"]');
        if (registrationButton) {
            setTimeout(() => {
                registrationButton.classList.add('animated');
            }, 100 + (registrationFormGroups.length * 50) + 50); // Opóźnij po ostatnim polu
        }
    }

});
