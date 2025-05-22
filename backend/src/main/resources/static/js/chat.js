document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded: chat.js loaded.');

    const chatButton = document.getElementById('chat-button');
    const chatWindow = document.getElementById('chat-window');
    const chatCloseButton = document.getElementById('chat-close-button');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input-field');
    const chatSendButton = document.getElementById('chat-send-button');

    if (!chatButton || !chatWindow || !chatCloseButton || !chatMessages || !chatInput || !chatSendButton) {
        console.log('One or more chat elements not found on this page. Chat functionality not initialized.');
        return;
    }

    console.log('Chat elements found. Initializing chat functionality.');

    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        messageElement.classList.add(sender);
        messageElement.textContent = text;
        chatMessages.appendChild(messageElement);

        // Przewiń na dół
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    function closeAnimateChatWindow() {
        // Usuń klasę, aby rozpocząć animację znikania (transition w CSS)
        chatWindow.classList.remove('is-open');
        console.log('Chat: Usunięto klasę is-open.');

        // Poczekaj na zakończenie animacji (długość transition w CSS)
        // Używamy setTimeout, pasującego do najdłuższej animacji (0.3s)
        const transitionDuration = 300; // Czas w milisekundach (0.3s z CSS)
        setTimeout(() => {
            chatWindow.style.display = 'none'; // Ukryj element fizycznie po animacji
            console.log('Chat: Ustawiono display: none po animacji.');
        }, transitionDuration);
    }


    chatButton.addEventListener('click', function() {
        chatWindow.style.display = 'flex'; // Najpierw ustaw display, aby element był w layoucie
        console.log('Chat: Ustawiono display okna na flex.');

        // Użyj requestAnimationFrame, aby upewnić się, że przeglądarka zarejestrowała display: flex
        // przed dodaniem klasy triggerującej transition
        requestAnimationFrame(() => {
            chatWindow.classList.add('is-open'); // Dodaj klasę, która uruchomi transition
            console.log('Chat: Dodano klasę is-open.');
        });


        chatButton.style.display = 'none'; // Ukryj przycisk czatu
        chatMessages.scrollTop = chatMessages.scrollHeight; // Przewiń na dół (jeśli są wiadomości)
        chatInput.focus(); // Ustaw fokus na polu wprowadzania
    });

    chatCloseButton.addEventListener('click', function() {
        closeAnimateChatWindow();
        chatButton.style.display = 'flex';
    });

    // --- Funkcja wysyłania wiadomości do backendu ---
    async function sendMessage() {
        const messageText = chatInput.value.trim();
        if (!messageText) {
            // Jeśli wiadomość jest pusta, nie rób nic
            return;
        }

        // Dodaj wiadomość użytkownika do okna czatu
        addMessage(messageText, 'user');

        // Wyczyść pole wprowadzania i zablokuj je tymczasowo
        chatInput.value = '';
        chatInput.disabled = true;
        chatSendButton.disabled = true;

        // Dodaj tymczasową wiadomość ładowania od bota
        addMessage('...', 'bot');

        // Pobranie tokenu CSRF z meta tagów (przy założeniu, że są dostępne)
        const csrfTokenMeta = document.querySelector('meta[name="_csrf_token"]');
        const csrfHeaderMeta = document.querySelector('meta[name="_csrf_headerName"]');
        const token = csrfTokenMeta ? csrfTokenMeta.content : null;
        const headerName = csrfHeaderMeta ? csrfHeaderMeta.content : null;

        const headers = {
            'Content-Type': 'application/json',
        };

        if (token && headerName) {
            headers[headerName] = token;
        } else {
            console.warn('CSRF token meta tags not found. Backend chat request may fail due to security constraints.');
        }

        try {
            const response = await fetch('/chat/send', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ message: messageText })
            });

            const botMessages = chatMessages.querySelectorAll('.chat-message.bot');
            const loadingMessage = botMessages.length > 0 && botMessages[botMessages.length - 1].innerText === '...' ? botMessages[botMessages.length - 1] : null;

            if (loadingMessage) {
                loadingMessage.remove();
            }

            if (!response.ok) {
                if (response.status === 403) {
                    addMessage("Błąd: Odmowa dostępu. Proszę odświeżyć stronę lub zalogować się ponownie, jeśli problem się powtórzy.", 'bot');
                    console.error('Backend chat failed with 403 Forbidden. Likely CSRF issue or authentication problem.');
                } else {
                    // Spróbuj odczytać tekst błędu z odpowiedzi (może być pusty)
                    const errorText = await response.text().catch(() => 'Nie udało się odczytać treści błędu.');
                    addMessage(`Błąd serwera: ${response.status}.`, 'bot'); // Możesz dodać errorText jeśli chcesz pokazać szczegóły błędu z backendu
                    console.error('Backend chat error response:', response.status, errorText);
                }

            } else {
                const result = await response.json();
                console.log('Backend response:', result);

                if (result && result.reply) {
                    addMessage(result.reply, 'bot');
                } else {
                    addMessage('Błąd: Serwer zwrócił nieznany format odpowiedzi.', 'bot');
                    console.error('Invalid response format from backend:', result);
                }
            }

        } catch (error) {
            console.error('Fetch or processing error:', error);

            const botMessages = chatMessages.querySelectorAll('.chat-message.bot');
            const loadingMessage = botMessages.length > 0 && botMessages[botMessages.length - 1].innerText === '...' ? botMessages[botMessages.length - 1] : null;

            if (loadingMessage) {
                loadingMessage.remove();
            }
            addMessage(`Wystąpił błąd: ${error.message}. Spróbuj ponownie.`, 'bot');

        } finally {
            chatInput.disabled = false;
            chatSendButton.disabled = false;
            chatInput.focus();
        }
    }

    chatSendButton.addEventListener('click', sendMessage);

    // Obsługa wysyłania wiadomości po naciśnięciu Enter w polu input
    chatInput.addEventListener('keypress', function(event) {
        // Sprawdź, czy naciśnięto klawisz Enter (kod 13 lub 'Enter')
        if (event.key === 'Enter') {
            event.preventDefault(); // Zapobiegaj domyślnej akcji formularza/wprowadzania (np. nowa linia)
            sendMessage(); // Wywołaj funkcję wysyłającą wiadomość
        }
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;

});