document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const navbarMenu = document.getElementById('navbarMenu');

    if (hamburgerBtn && navbarMenu) {
        hamburgerBtn.addEventListener('click', function() {
            navbarMenu.classList.toggle('show');
        });
    }

    // Services Dropdown Toggle
    const servicesBtn = document.getElementById('servicesBtn');
    const servicesDropdown = document.getElementById('servicesDropdown');

    if (servicesBtn && servicesDropdown) {
        servicesBtn.addEventListener('click', function(e) {
            e.preventDefault();
            servicesBtn.classList.toggle('active');
            servicesDropdown.classList.toggle('show');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.navbar-services') && servicesDropdown.classList.contains('show')) {
                servicesDropdown.classList.remove('show');
                servicesBtn.classList.remove('active');
            }
        });
    }

    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Form field animations
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

    // Add page transition effect
    const pageContent = document.querySelector('.page-content');
    if (pageContent) {
        pageContent.classList.add('fade-in');
    }

    // Automatically hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-danger, .alert-info');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease-out';
            alert.style.opacity = '0';

            setTimeout(() => {
                alert.style.display = 'none';
            }, 500);
        }, 5000);
    });

    // Add result animation
    const resultElement = document.getElementById('result');
    if (resultElement && resultElement.innerHTML.trim() !== '') {
        resultElement.classList.add('fade-in');
    }

    // Add active class to current nav link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-menu a');

    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

// Form submission with loading state
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('blood-form');

    if (form) {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                // Save original text
                const originalText = submitButton.innerText;

                // Change to loading state
                submitButton.innerText = 'Przetwarzanie...';
                submitButton.disabled = true;
                submitButton.classList.add('loading');

                // Reset button state after form submission
                // This timeout is for visual effect, as the actual form submission 
                // will navigate away or be handled by the form's submit handler
                setTimeout(() => {
                    submitButton.innerText = originalText;
                    submitButton.disabled = false;
                    submitButton.classList.remove('loading');
                }, 10000); // Failsafe timeout in case the form submission is not properly handled
            }
        });
    }
});
