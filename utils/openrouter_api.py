import os
import json
import logging
import requests
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Get API key from environment variables with fallback
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"  # Darmowy model Mistral

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://cv-optimizer-pro.repl.co/"  # Replace with your actual domain
}

def send_api_request(prompt, max_tokens=2000, retry_count=1, retry_delay=1):
    """
    Send a request to the OpenRouter API with minimal retry logic

    Args:
        prompt (str): The prompt to send to the AI
        max_tokens (int): Maximum number of tokens in the response
        retry_count (int): Number of retry attempts for rate limiting errors
        retry_delay (int): Delay in seconds between retries

    Returns:
        str: The AI-generated response or a fallback message if all retries fail
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found")
        raise ValueError("OpenRouter API key not set in environment variables")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert resume editor and career advisor. Always respond in the same language as the CV or job description provided by the user."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }

    try:
        logger.debug(f"Sending request to OpenRouter API")
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=5)

        # Check specifically for rate limiting error
        if response.status_code == 429:
            logger.error("Rate limit hit from OpenRouter API")
            # W przypadku ograniczenia zapytań, zwracamy przygotowany komunikat bez ponownych prób
            return "[RATE_LIMITED] Przekroczono limit zapytań API. Proszę spróbować ponownie za kilka minut."

        # For other errors, raise immediately
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Received response from OpenRouter API")

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logger.error("Unexpected API response format")
            return "[ERROR] Nieoczekiwany format odpowiedzi API."

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if "429" in str(e):
            return "[RATE_LIMITED] Przekroczono limit zapytań API. Proszę spróbować ponownie za kilka minut."
        return f"[ERROR] Błąd komunikacji z API: {str(e)}"

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing API response: {str(e)}")
        return f"[ERROR] Błąd przetwarzania odpowiedzi API: {str(e)}"

def detect_seniority_level(cv_text, job_description):
    """
    Detect seniority level (junior, mid, senior) based on CV and job description
    """
    prompt = f"""
    TASK: Określ poziom seniority (junior, mid, senior) na podstawie CV i opisu stanowiska.

    Wskazówki do analizy:

    1. Sprawdź lata doświadczenia w CV
    2. Przeanalizuj poziom odpowiedzialności w poprzednich rolach
    3. Oceń wymagania z opisu stanowiska
    4. Porównaj umiejętności z CV z wymaganiami w opisie stanowiska

    Zwróć tylko jeden z poniższych poziomów:
    - "junior" - dla początkujących specjalistów z doświadczeniem 0-2 lata
    - "mid" - dla specjalistów z doświadczeniem 2-5 lat
    - "senior" - dla ekspertów z doświadczeniem 5+ lat

    CV:
    {cv_text[:2000]}...

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem: junior, mid lub senior.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()

        if response in ["junior", "mid", "senior"]:
            return response
        else:
            # Domyślnie zwróć mid-level jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid seniority level detected: {response}. Using 'mid' as default.")
            return "mid"
    except Exception as e:
        logger.error(f"Error detecting seniority level: {str(e)}")
        return "mid"  # Domyślny poziom

def detect_job_type(job_description):
    """
    Detect job type (physical, technical, office) based on job description
    """
    prompt = f"""
    TASK: Określ typ pracy opisanej w ogłoszeniu o pracę.

    Możliwe typy pracy:
    - "physical" - praca fizyczna (np. kierowca, magazynier, pracownik produkcji)
    - "technical" - praca techniczna (np. mechanik, elektryk, technik)
    - "office" - praca biurowa (np. administrator, asystent, koordynator)
    - "professional" - praca specjalistyczna (np. lekarz, prawnik, nauczyciel)
    - "creative" - praca kreatywna (np. grafik, projektant, artysta)
    - "it" - praca w IT (np. programista, administrator sieci, analityk danych)

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem - kod typu pracy.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()

        valid_job_types = ["physical", "technical", "office", "professional", "creative", "it"]

        if response in valid_job_types:
            return response
        else:
            # Domyślnie zwróć office jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid job type detected: {response}. Using 'office' as default.")
            return "office"
    except Exception as e:
        logger.error(f"Error detecting job type: {str(e)}")
        return "office"  # Domyślny typ pracy

def detect_specific_role(job_description):
    """
    Detect specific job role based on job description
    """
    prompt = f"""
    TASK: Określ konkretną rolę zawodową opisaną w ogłoszeniu o pracę.

    Wybierz jedną konkretną rolę, która najlepiej pasuje do opisu, na przykład:
    - kierowca
    - magazynier
    - sprzedawca
    - księgowy
    - programista
    - nauczyciel
    - lekarz
    - grafik
    - mechanik
    - inżynier

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem - nazwa konkretnej roli zawodowej, bez żadnych dodatkowych słów.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        return response.strip().lower()
    except Exception as e:
        logger.error(f"Error detecting specific role: {str(e)}")
        return "specjalista"  # Domyślna rola

def detect_industry(job_description):
    """
    Detect industry based on job description
    """
    prompt = f"""
    TASK: Określ branżę na podstawie opisu stanowiska.

    Możliwe branże:
    - "it" - technologia, programowanie, analiza danych, IT
    - "finance" - finanse, bankowość, księgowość, ubezpieczenia
    - "marketing" - marketing, reklama, PR, social media
    - "healthcare" - służba zdrowia, farmacja, medycyna
    - "hr" - HR, rekrutacja, zasoby ludzkie
    - "education" - edukacja, szkolnictwo, e-learning
    - "engineering" - inżynieria, produkcja, budownictwo
    - "transport" - transport, logistyka, spedycja
    - "retail" - handel detaliczny, sprzedaż, obsługa klienta
    - "legal" - prawo, usługi prawne
    - "creative" - kreatywna, design, sztuka, UX/UI
    - "general" - inna branża lub brak wyraźnej specjalizacji

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem - kod branży.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()

        valid_industries = ["it", "finance", "marketing", "healthcare", "hr", 
                           "education", "engineering", "transport", "retail",
                           "legal", "creative", "general"]

        if response in valid_industries:
            return response
        else:
            # Domyślnie zwróć general jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid industry detected: {response}. Using 'general' as default.")
            return "general"
    except Exception as e:
        logger.error(f"Error detecting industry: {str(e)}")
        return "general"  # Domyślna branża

def get_role_specific_competencies(role):
    """
    Get role-specific competencies, certifications and typical achievements
    """
    role_competencies = {
        "kierowca": {
            "certifications": [
                "Prawo jazdy kategorii B/C/C+E/D",
                "Karta kierowcy",
                "Świadectwo kwalifikacji zawodowej",
                "Zaświadczenie o niekaralności",
                "Certyfikat ADR (przewóz materiałów niebezpiecznych)",
                "Uprawnienia HDS (hydrauliczny dźwig samochodowy)"
            ],
            "skills": [
                "Znajomość przepisów ruchu drogowego",
                "Obsługa tachografu cyfrowego",
                "Planowanie optymalnych tras",
                "Dbałość o stan techniczny pojazdu",
                "Zabezpieczanie ładunku",
                "Prowadzenie dokumentacji transportowej",
                "Obsługa GPS i systemów nawigacyjnych",
                "Podstawowa znajomość mechaniki pojazdowej"
            ],
            "achievements": [
                "Przejechanych X kilometrów bez wypadku",
                "Utrzymanie zużycia paliwa X% poniżej średniej firmowej",
                "Terminowość dostaw na poziomie X%",
                "Skrócenie czasu dostawy o X% dzięki optymalizacji trasy",
                "Bezbłędne prowadzenie dokumentacji przez X miesięcy",
                "Realizacja X dostaw miesięcznie",
                "Obsługa X stałych klientów z najwyższymi ocenami satysfakcji"
            ],
            "language_style": "Profesjonalny, konkretny, z naciskiem na bezpieczeństwo i odpowiedzialność"
        },
        "magazynier": {
            "certifications": [
                "Uprawnienia na wózki widłowe",
                "Uprawnienia na suwnice",
                "Certyfikat BHP",
                "Uprawnienia do obsługi systemów WMS"
            ],
            "skills": [
                "Obsługa skanerów i czytników kodów",
                "Kompletacja zamówień",
                "Inwentaryzacja",
                "Obsługa systemów magazynowych (WMS)",
                "Przyjmowanie i wydawanie towaru",
                "Kontrola jakościowa produktów",
                "Pakowanie i zabezpieczanie towaru"
            ],
            "achievements": [
                "Zwiększenie wydajności kompletacji o X%",
                "Zmniejszenie błędów w zamówieniach o X%",
                "Wdrożenie usprawnień w procesie X, co skróciło czas o Y%",
                "Bezbłędne przeprowadzenie X inwentaryzacji",
                "Utrzymanie 100% dokładności w zarządzaniu zapasami przez X miesięcy",
                "Obsługa X palet dziennie"
            ],
            "language_style": "Precyzyjny, operacyjny, podkreślający dokładność i efektywność"
        },
        "programista": {
            "certifications": [
                "Certyfikaty Microsoft/AWS/Google Cloud",
                "Certyfikaty językowe (Java, Python)",
                "Certyfikaty Agile/Scrum",
                "Certyfikat ITIL",
                "Certyfikat cyberbezpieczeństwa"
            ],
            "skills": [
                "Znajomość technologii X, Y, Z",
                "Tworzenie czystego, testowalnego kodu",
                "Projektowanie architektury systemów",
                "Testowanie i debugowanie aplikacji",
                "Praca z systemami kontroli wersji",
                "Współpraca w zespole programistycznym",
                "Code review",
                "Ciągła integracja/wdrażanie (CI/CD)"
            ],
            "achievements": [
                "Optymalizacja wydajności systemu X o Y%",
                "Skrócenie czasu ładowania aplikacji o X%",
                "Zredukowanie liczby błędów o X% poprzez wdrożenie testów automatycznych",
                "Wdrożenie X nowych funkcjonalności w ciągu Y miesięcy",
                "Przeprowadzenie refaktoryzacji kodu, co zmniejszyło jego złożoność o X%",
                "Stworzenie rozwiązania, które zaoszczędziło firmie X zł rocznie"
            ],
            "language_style": "Techniczny, analityczny, z wykorzystaniem specjalistycznej terminologii IT"
        },
        "sprzedawca": {
            "certifications": [
                "Certyfikat obsługi klienta",
                "Certyfikat sprzedażowy",
                "Uprawnienia do obsługi kasy fiskalnej",
                "Certyfikat z technik negocjacji"
            ],
            "skills": [
                "Profesjonalna obsługa klienta",
                "Znajomość technik sprzedaży",
                "Obsługa kasy fiskalnej i terminali płatniczych",
                "Zarządzanie zapasami na półkach",
                "Przygotowywanie ekspozycji produktów",
                "Rozwiązywanie problemów klientów",
                "Realizacja planów sprzedażowych"
            ],
            "achievements": [
                "Przekroczenie celu sprzedażowego o X%",
                "Zwiększenie średniej wartości koszyka o X%",
                "Pozyskanie X nowych stałych klientów",
                "Utrzymanie najwyższego wskaźnika satysfakcji klienta przez X miesięcy",
                "Przeprowadzenie X skutecznych akcji promocyjnych",
                "Wzrost sprzedaży w kategorii X o Y%"
            ],
            "language_style": "Nastawiony na klienta, entuzjastyczny, przekonujący"
        }
    }

    # Dodaj więcej ról w miarę potrzeb

    # Jeśli rola nie jest zdefiniowana, zwróć ogólne kompetencje
    if role not in role_competencies:
        return {
            "certifications": ["Certyfikaty branżowe", "Szkolenia specjalistyczne"],
            "skills": ["Umiejętności interpersonalne", "Organizacja pracy", "Rozwiązywanie problemów"],
            "achievements": ["Przekroczenie celów o X%", "Optymalizacja procesów", "Realizacja projektów"],
            "language_style": "Profesjonalny, rzeczowy, zorientowany na wyniki"
        }

    return role_competencies[role]

def get_job_type_template(job_type):
    """
    Get CV template guidance based on job type
    """
    templates = {
        "physical": """
    STRUKTURA CV DLA PRACY FIZYCZNEJ:
    1. Dane kontaktowe na górze strony
    2. Krótkie podsumowanie zawodowe (3-4 zdania)
    3. Uprawnienia i certyfikaty (na pierwszym miejscu)
    4. Doświadczenie zawodowe (konkretne dane liczbowe)
    5. Wykształcenie i kursy (zwięźle)
    6. Umiejętności techniczne i praktyczne (z podziałem na kategorie)

    FORMAT:
    - Maksymalnie 1-2 strony
    - Proste, czytelne formatowanie
    - Wypunktowania zamiast długich paragrafów
    - Podkreślenie uprawnień i kwalifikacji zawodowych

    STYL JĘZYKOWY:
    - Konkretne, rzeczowe sformułowania
    - Proste zdania bez żargonu
    - Nacisk na praktyczne umiejętności
    - Używanie czasowników czynnościowych (obsługiwałem, dostarczałem, naprawiałem)
    """,
        "technical": """
    STRUKTURA CV DLA PRACY TECHNICZNEJ:
    1. Dane kontaktowe i dane osobowe
    2. Podsumowanie zawodowe z kluczowymi umiejętnościami
    3. Kwalifikacje techniczne i uprawnienia
    4. Doświadczenie zawodowe z konkretnymi projektami
    5. Wykształcenie i specjalistyczne szkolenia
    6. Umiejętności techniczne z poziomem zaawansowania

    FORMAT:
    - 1-2 strony
    - Przejrzyste sekcje z podtytułami
    - Używanie tabel dla umiejętności technicznych
    - Uwypuklenie certyfikatów i uprawnień

    STYL JĘZYKOWY:
    - Precyzyjny, techniczny język
    - Szczegółowy opis umiejętności
    - Używanie branżowej terminologii
    - Konkretne osiągnięcia z liczbami i parametrami
    """,
        "office": """
    STRUKTURA CV DLA PRACY BIUROWEJ:
    1. Dane kontaktowe i profesjonalny profil
    2. Zwięzłe podsumowanie zawodowe
    3. Doświadczenie zawodowe (chronologicznie)
    4. Umiejętności biurowe i znajomość oprogramowania
    5. Wykształcenie i kursy
    6. Osiągnięcia i dodatkowe kwalifikacje

    FORMAT:
    - 1-2 strony
    - Eleganckie, czyste formatowanie
    - Spójne czcionki i marginesy
    - Umiarkowane używanie kolorów

    STYL JĘZYKOWY:
    - Profesjonalny, biznesowy język
    - Użycie czasowników biznesowych (koordynowałem, zarządzałem, analizowałem)
    - Podkreślenie umiejętności organizacyjnych i komunikacyjnych
    - Formalne, ale przystępne sformułowania
    """,
        "professional": """
    STRUKTURA CV DLA PRACY SPECJALISTYCZNEJ:
    1. Dane kontaktowe i profesjonalny profil
    2. Podsumowanie ekspertyz i kluczowych kompetencji
    3. Doświadczenie zawodowe z podkreśleniem osiągnięć
    4. Wykształcenie, specjalizacje i certyfikacje
    5. Publikacje, projekty badawcze lub specjalistyczne osiągnięcia
    6. Umiejętności specjalistyczne i znajomość metodologii

    FORMAT:
    - 2-3 strony
    - Profesjonalne, uporządkowane formatowanie
    - Możliwość dodania sekcji publikacji/projektów
    - Hierarchiczna organizacja informacji

    STYL JĘZYKOWY:
    - Zaawansowany, specjalistyczny język
    - Terminologia branżowa na wysokim poziomie
    - Podkreślenie ekspertyzy i autorytetu w dziedzinie
    - Uwypuklenie wartości dodanej dla organizacji
    """,
        "creative": """
    STRUKTURA CV DLA PRACY KREATYWNEJ:
    1. Dane kontaktowe i link do portfolio
    2. Kreatywne, wyróżniające się podsumowanie
    3. Wybrane projekty i osiągnięcia (przed doświadczeniem)
    4. Doświadczenie zawodowe
    5. Umiejętności kreatywne i techniczne
    6. Wykształcenie i rozwój kreatywny

    FORMAT:
    - 1-2 strony, ale z wyróżniającym się designem
    - Możliwość niestandardowego układu
    - Elementy graficzne podkreślające kreatywność
    - Więcej swobody w kolorach i formatowaniu

    STYL JĘZYKOWY:
    - Dynamiczny, kreatywny język
    - Balans między profesjonalizmem a kreatywnością
    - Uwypuklenie procesów kreatywnych i wyników
    - Unikalny, osobisty ton głosu
    """,
        "it": """
    STRUKTURA CV DLA PRACY W IT:
    1. Dane kontaktowe i linki (GitHub, LinkedIn)
    2. Zwięzłe podsumowanie techniczne
    3. Umiejętności techniczne pogrupowane według kategorii
    4. Doświadczenie zawodowe z konkretnymi projektami
    5. Wykształcenie i certyfikaty techniczne
    6. Projekty osobiste i open source

    FORMAT:
    - 1-2 strony
    - Techniczne, przejrzyste formatowanie
    - Tabele lub paski postępu dla umiejętności
    - Elementy kodu/pseudokodu jako akcenty

    STYL JĘZYKOWY:
    - Techniczny, precyzyjny język
    - Używanie terminologii IT
    - Konkretne metryki i rezultaty techniczne
    - Podkreślenie znajomości technologii i rozwiązanych problemów
    """
    }

    return templates.get(job_type, templates["office"])

def get_industry_specific_prompt(industry, seniority, job_type=None, specific_role=None):
    """
    Get industry-specific prompt guidance, enhanced with job type and role specifics
    """
    # Domyślne wskazówki dla ogólnej branży
    industry_guidance = """
    - Użyj uniwersalnego języka biznesowego
    - Podkreśl umiejętności interpersonalne i adaptacyjne
    - Skup się na osiągnięciach mierzalnych w różnych kontekstach
    - Podkreśl znajomość standardowych narzędzi biznesowych
    """

    # Branżowo-specyficzne wskazówki
    industry_prompts = {
        "it": """
    - Użyj technicznych terminów branżowych i nazw technologii
    - Wymień konkretne języki programowania, narzędzia, frameworki z określeniem poziomu biegłości
    - Podkreśl umiejętność rozwiązywania złożonych problemów technicznych
    - Uwzględnij metodyki wytwarzania oprogramowania (np. Agile, Scrum)
    - Wykorzystaj mierzalne wskaźniki techniczne (optymalizacja wydajności, redukcja błędów)
    - Uwzględnij projekty open source i repozytoria kodu (GitHub, GitLab)
    """,
        "finance": """
    - Zastosuj precyzyjny język finansowy i terminologię branżową
    - Podkreśl umiejętności analityczne i znajomość regulacji (np. MSSF, US GAAP)
    - Uwzględnij konkretne wyniki finansowe i optymalizacje kosztów w procentach
    - Wyeksponuj znajomość systemów finansowych i umiejętność analizy danych
    - Podkreśl dokładność i dbałość o szczegóły w kontekście finansowym
    """,
        "marketing": """
    - Użyj dynamicznego, kreatywnego języka z branżowym słownictwem marketingowym
    - Podaj konkretne wyniki kampanii (ROI, conversion rate, zasięg)
    - Wymień znajomość platform marketingowych i narzędzi analitycznych
    - Podkreśl umiejętności w zakresie content marketingu i mediów społecznościowych
    - Uwzględnij kreatywne projekty i case studies z mierzalnymi efektami
    """,
        "healthcare": """
    - Zastosuj profesjonalną terminologię medyczną
    - Podkreśl certyfikaty i uprawnienia branżowe
    - Wyeksponuj znajomość procedur medycznych i regulacji (np. RODO w kontekście danych medycznych)
    - Uwzględnij doświadczenie z konkretną aparaturą medyczną lub systemami opieki zdrowotnej
    - Podkreśl umiejętności interpersonalne w kontekście opieki nad pacjentem
    """,
        "hr": """
    - Zastosuj terminologię HR i zarządzania talentami
    - Podaj konkretne dane dotyczące rekrutacji, retencji i rozwoju pracowników
    - Podkreśl znajomość prawa pracy i systemów HR
    - Uwzględnij zrealizowane projekty rozwojowe i ich wpływ na organizację
    - Wyeksponuj umiejętności miękkie i komunikacyjne
    """,
        "education": """
    - Użyj terminologii edukacyjnej i pedagogicznej
    - Podkreśl certyfikaty nauczycielskie i metody edukacyjne
    - Uwzględnij opracowane materiały dydaktyczne i programy nauczania
    - Wyeksponuj mierzalne wyniki edukacyjne uczniów/studentów
    - Podkreśl umiejętności dydaktyczne i zarządzania klasą/grupą
    """,
        "engineering": """
    - Zastosuj precyzyjny język inżynieryjny i techniczny
    - Wymień konkretne projekty inżynieryjne z parametrami technicznymi
    - Podkreśl znajomość norm i standardów branżowych
    - Uwzględnij optymalizacje procesów i oszczędności materiałowe/czasowe
    - Wyeksponuj umiejętność rozwiązywania złożonych problemów technicznych
    """,
        "transport": """
    - Zastosuj precyzyjny język transportowy i logistyczny
    - Podkreśl znajomość przepisów transportowych i dokumentacji
    - Uwzględnij konkretne dane dotyczące realizowanych tras, ładunków, kilometrażu
    - Wyeksponuj znajomość procedur bezpieczeństwa i efektywności transportu
    - Podkreśl osiągnięcia w zakresie terminowości i jakości dostaw
    """,
        "retail": """
    - Użyj języka zorientowanego na klienta i sprzedaż
    - Podaj konkretne wyniki sprzedażowe i wskaźniki KPI
    - Wymień znajomość systemów kasowych i zarządzania zapasami
    - Podkreśl umiejętności w zakresie merchandisingu i układania ekspozycji
    - Uwzględnij osiągnięcia w zakresie obsługi klienta i rozwiązywania problemów
    """,
        "legal": """
    - Zastosuj precyzyjny język prawniczy i formalny styl
    - Podkreśl znajomość konkretnych aktów prawnych i orzecznictwa
    - Uwzględnij prowadzone sprawy/projekty z zachowaniem poufności
    - Wyeksponuj umiejętności analityczne i interpretacyjne
    - Podkreśl certyfikaty i uprawnienia prawnicze
    """,
        "creative": """
    - Użyj kreatywnego, dynamicznego języka
    - Uwzględnij portfolio projektów kreatywnych z konkretnymi efektami
    - Podkreśl znajomość narzędzi projektowych i technologii kreatywnych
    - Wyeksponuj umiejętność pracy w zespołach interdyscyplinarnych
    - Podkreśl nagrody i wyróżnienia w dziedzinach kreatywnych
    """
    }

    # Pobierz wskazówki dla konkretnej branży lub użyj domyślnych
    if industry in industry_prompts:
        industry_guidance = industry_prompts[industry]

    # Modyfikacje pod kątem seniority
    seniority_guidance = {
        "junior": """
    - Podkreśl zapał do nauki i szybkiego przyswajania wiedzy
    - Uwypuklij projekty szkolne/akademickie i ich praktyczne zastosowanie
    - Skup się na potencjale i umiejętnościach podstawowych
    - Pokaż gotowość do rozwoju pod mentorskim okiem
    """,
        "mid": """
    - Zbalansuj doświadczenie z potencjałem rozwojowym
    - Podkreśl samodzielnie zrealizowane projekty i ich efekty
    - Uwypuklij specjalizacje i konkretne obszary ekspertyzy
    - Pokaż umiejętność współpracy z różnymi interesariuszami
    """,
        "senior": """
    - Uwypuklij strategiczne myślenie i szerszą perspektywę biznesową
    - Podkreśl role przywódcze i mentorskie
    - Skup się na długofalowych efektach i transformacyjnych projektach
    - Pokaż umiejętność kierowania zespołami i zarządzania zasobami
    - Uwzględnij wpływ na działalność biznesową i KPI organizacji
    """
    }

    # Dodaj wskazówki dotyczące typu pracy
    job_type_guidance = ""
    if job_type:
        job_type_template = get_job_type_template(job_type)
        job_type_guidance = f"\n\nWSKAZÓWKI DOTYCZĄCE TYPU PRACY ({job_type.upper()}):\n{job_type_template}"

    # Dodaj wskazówki dotyczące konkretnej roli
    role_guidance = ""
    if specific_role:
        competencies = get_role_specific_competencies(specific_role)

        role_guidance = f"\n\nWYMAGANE KOMPETENCJE DLA ROLI: {specific_role.upper()}\n"

        # Certyfikaty i uprawnienia
        role_guidance += "\nSugerowane certyfikaty i uprawnienia:\n"
        for cert in competencies.get("certifications", []):
            role_guidance += f"- {cert}\n"

        # Umiejętności
        role_guidance += "\nKluczowe umiejętności dla tej roli:\n"
        for skill in competencies.get("skills", []):
            role_guidance += f"- {skill}\n"

        # Typowe osiągnięcia
        role_guidance += "\nTypowe osiągnięcia w tej roli (zamień X, Y, Z na realne liczby):\n"
        for achievement in competencies.get("achievements", []):
            role_guidance += f"- {achievement}\n"

        # Styl języka
        role_guidance += f"\nSugerowany styl języka: {competencies.get('language_style', 'Profesjonalny')}\n"

    result = industry_guidance + "\n" + seniority_guidance.get(seniority, seniority_guidance["mid"])

    # Dodaj dodatkowe wskazówki, jeśli są dostępne
    if job_type_guidance:
        result += job_type_guidance
    if role_guidance:
        result += role_guidance

    return result

def get_measurable_achievements_prompt(seniority):
    """
    Get prompt to encourage adding measurable achievements based on seniority
    """
    prompts = {
        "junior": """
    Nawet dla juniora dodaj mierzalne osiągnięcia: 
    - Jeśli brak konkretnych liczb w CV, dodaj przybliżone wyniki: "przygotowałem około 10 analiz", "wsparłem X projektów"
    - Zamień ogólne stwierdzenia na konkretne: "nauczyłem się X technologii w ciągu 3 miesięcy"
    - Uwzględnij efekty edukacyjne: "ukończyłem studia z wynikiem X% / w czołówce Y% studentów"
    - Dodaj wyniki projektów studenckich/hobbyistycznych z konkretnymi liczbami
    """,
        "mid": """
    Wzbogać CV o konkretne, mierzalne wyniki:
    - Dodaj procenty poprawy procesów: "zwiększyłem wydajność o X%", "skróciłem czas realizacji o Y dni"
    - Uwzględnij konkretne wskaźniki: "przeprowadziłem X kampanii", "wdrożyłem Y funkcjonalności"
    - Zamień ogólniki na liczby: "zarządzałem 5-osobowym zespołem", "pozyskałem X nowych klientów"
    - Dodaj skalę projektów: "projekt o budżecie X zł", "system dla Y użytkowników"
    """,
        "senior": """
    Umieść strategiczne, biznesowe mierzalne osiągnięcia:
    - Dodaj wskaźniki finansowe: "zwiększyłem przychody o X%", "zredukowałem koszty o Y zł"
    - Uwzględnij wpływ na organizację: "wdrożyłem strategię, która zwiększyła rentowność o X%"
    - Podkreśl efekty przywództwa: "kierowałem zespołem X osób, osiągając Y% wzrostu produktywności"
    - Zamień każde ogólnikowe osiągnięcie na konkretne z liczbami, procentami i skalą czasową
    - Uwzględnij wyniki transformacji: "przeprowadziłem restrukturyzację działu X, co przyniosło Y oszczędności"
    """
    }

    return prompts.get(seniority, prompts["mid"])

def get_structural_quality_control_prompt(seniority, industry):
    """
    Get structural quality control prompt based on seniority and industry
    """
    base_prompt = """
    Zapewnij optymalną strukturę CV:
    - Akapity nie dłuższe niż 3-4 linijki tekstu
    - Każde doświadczenie zawodowe opisane w 3-5 punktach
    - Sekcja umiejętności podzielona na kategorie
    - Zachowaj spójny format dla dat i lokalizacji
    - Stosuj nagłówki w standardzie ATS
    """

    industry_specific = {
        "it": """
    - Dodaj sekcję umiejętności technicznych na początku, kategoryzując je
    - Dla każdej technologii określ poziom zaawansowania (%)
    - Używaj wypunktowań dla osiągnięć technicznych (4-6 punktów na rolę)
    - Skróć historię zawodową do najważniejszych technologicznie stanowisk
    """,
        "finance": """
    - Użyj precyzyjnych nagłówków sekcji (np. "Doświadczenie w księgowości zarządczej")
    - Każdy punkt osiągnięć powinien mieć aspekt ilościowy
    - Struktura punktów: działanie, sposób, rezultat, skala
    - Zachowaj formalny układ bez elementów kreatywnych
    """,
        "marketing": """
    - Użyj kreatywnych, ale jasnych nagłówków sekcji
    - Każde doświadczenie zawodowe opisz w 4-6 punktach
    - Zrównoważ aspekty kreatywne i analityczne w punktach
    - Dodaj sekcję z przykładami kampanii/projektów
    """,
        "creative": """
    - Zastosuj przejrzysty układ podkreślający portfolio
    - Punkty osiągnięć skup na efekcie i procesie kreatywnym
    - Używaj dynamicznych czasowników na początku punktów
    - Zrównoważ techniczne aspekty z kreatywnymi
    """
    }

    language_style = {
        "junior": """
    - Użyj prostego, bezpośredniego języka
    - Stosuj podstawową terminologię branżową
    - Unikaj zaawansowanego słownictwa
    - Podkreślaj entuzjazm i potencjał
    """,
        "mid": """
    - Zbalansuj profesjonalny język z przystępnością
    - Stosuj branżowe terminy w kontekście
    - Unikaj zbyt ogólnikowych stwierdzeń
    - Zachowaj spójność stylu w całym dokumencie
    """,
        "senior": """
    - Stosuj zaawansowany język biznesowy i branżowy
    - Używaj precyzyjnych terminów strategicznych
    - Podkreślaj aspekty przywódcze w stylu komunikacji
    - Zachowaj profesjonalny, pewny siebie ton
    """
    }

    industry_guidance = industry_specific.get(industry, "")
    style_guidance = language_style.get(seniority, language_style["mid"])

    return base_prompt + "\n" + industry_guidance + "\n" + style_guidance

def optimize_cv_with_keywords(cv_text, job_description, keywords_data=None):
    """
    Create an optimized version of CV with advanced skills extraction, market analysis,
    and AI-powered career path optimization. Uses advanced NLP to identify and highlight
    transferable skills while maintaining strict truthfulness.
    """

    # Analiza wstępna CV i wymagań
    try:
        seniority = detect_seniority_level(cv_text, job_description)
        industry = detect_industry(job_description)
        job_type = detect_job_type(job_description)
        specific_role = detect_specific_role(job_description)
    except Exception as e:
        logger.error(f"Error in initial analysis: {str(e)}")
        seniority, industry, job_type, specific_role = "mid", "general", "office", "specialist"

    warning_prompt = """
    ZAAWANSOWANA ANALIZA I OPTYMALIZACJA CV:

    1. ANALIZA ZGODNOŚCI:
       - Identyfikacja kluczowych wymagań stanowiska
       - Mapowanie umiejętności kandydata do wymagań
       - Obliczenie % dopasowania do stanowiska
       - Sugestie uzupełnienia brakujących umiejętności

    2. OPTYMALIZACJA DOŚWIADCZENIA:
       - Identyfikacja projektów o największym znaczeniu
       - Kwantyfikacja osiągnięć (%, liczby, skala)
       - Wydobycie ukrytych umiejętności z projektów
       - Podkreślenie transferowalnych kompetencji

    3. ANALIZA RYNKOWA:
       - Porównanie z aktualnymi trendami branżowymi
       - Identyfikacja unikalnych atutów kandydata
       - Sugestie rozwoju zgodne z trendami rynku
       - Analiza konkurencyjności profilu

    4. PERSONALIZACJA I FORMATOWANIE:
       - Dostosowanie języka do kultury firmy
       - Optymalizacja pod kątem ATS
       - Hierarchizacja treści wg znaczenia
       - Profesjonalne formatowanie sekcji

    KRYTYCZNE ZASADY PRAWDZIWOŚCI:
    """
    # Wykryj poziom stanowiska i branżę
    try:
        seniority = detect_seniority_level(cv_text, job_description)
        industry = detect_industry(job_description)
        job_type = detect_job_type(job_description)
        specific_role = detect_specific_role(job_description)
    except Exception as e:
        logger.error(f"Error detecting context: {str(e)}")
        seniority, industry, job_type, specific_role = "junior", "general", "office", "specjalista"

    prompt = f"""
    TASK: Przeprowadź ostrożną analizę CV nauczyciela i pokaż, jak jego/jej rzeczywiste umiejętności i doświadczenie mogą być wartościowe w roli przedstawiciela handlowego. 

    KLUCZOWE ZASADY:
    1. Zachowaj wszystkie prawdziwe informacje z CV
    2. Nie wymyślaj żadnego doświadczenia w sprzedaży
    3. Pokaż, jak umiejętności nauczyciela przekładają się na sprzedaż:
       - Umiejętności prezentacji
       - Komunikacja z różnymi osobami
       - Cierpliwość w wyjaśnianiu
       - Zdolność przekonywania
       - Organizacja pracy
    4. Uczciwie przedstaw brak doświadczenia w sprzedaży jako potencjał do nauki

    KLUCZOWE OBSZARY ANALIZY:

    1. UMIEJĘTNOŚCI TRANSFEROWALNE:
       - Przeanalizuj każde doświadczenie pod kątem umiejętności, które można przenieść na nowe stanowisko
       - Znajdź powiązania między obecnymi umiejętnościami a wymaganiami stanowiska
       - Przekształć ogólne umiejętności w konkretne atuty dla nowej roli

    2. UKRYTE KOMPETENCJE:
       - Zidentyfikuj umiejętności wynikające z hobby, wolontariatu, projektów osobistych
       - Wydobądź kompetencje z pozornie niezwiązanych doświadczeń
       - Znajdź nietypowe źródła wartościowych umiejętności

    3. POTENCJAŁ ROZWOJOWY:
       - Podkreśl szybkość uczenia się i adaptacji
       - Zaznacz wszystkie kursy, szkolenia i certyfikaty
       - Uwypuklij chęć rozwoju w nowym kierunku

    4. OSIĄGNIĘCIA I SUKCESY:
       - Przekształć każde osiągnięcie w kontekst nowego stanowiska
       - Podkreśl uniwersalne sukcesy (np. praca zespołowa, efektywność)
       - Kwantyfikuj osiągnięcia tam, gdzie to możliwe

    WSKAZÓWKI SPECJALNE:

    1. Dla kandydata bez doświadczenia w {specific_role}:
       - Znajdź wszystkie sytuacje wymagające podobnych umiejętności
       - Przekształć doświadczenia z innych obszarów na język nowej roli
       - Podkreśl projekty osobiste lub akademickie związane ze stanowiskiem

    2. Transformacja umiejętności:
       - Komunikacja → Obsługa klienta / Prezentacje / Negocjacje
       - Organizacja → Zarządzanie projektami / Koordynacja
       - Analiza → Rozwiązywanie problemów / Optymalizacja procesów

    3. Wydobycie wartości z każdego doświadczenia:
       - Praca w restauracji → Obsługa klienta, praca pod presją czasu, zarządzanie priorytetami
       - Sprzedaż detaliczna → Negocjacje, prezentacja produktu, realizacja celów
       - Projekty studenckie → Zarządzanie projektami, praca zespołowa, dotrzymywanie terminów

    STRUKTURA NOWEGO CV:

    1. Silne podsumowanie zawodowe:
       - Podkreśl główne atuty pasujące do stanowiska
       - Zaznacz gotowość do szybkiego przyswajania wiedzy
       - Pokaż entuzjazm i motywację do rozwoju w nowym kierunku

    2. Kluczowe umiejętności:
       - Podziel na sekcje odpowiadające wymaganiom stanowiska
       - Dodaj poziom zaawansowania i kontekst zdobycia umiejętności
       - Uwzględnij umiejętności miękkie istotne dla roli

    3. Doświadczenie zawodowe:
       - Przekształć każde doświadczenie pod kątem nowej roli
       - Podkreśl osiągnięcia związane z wymaganymi kompetencjami
       - Używaj języka branżowego dopasowanego do stanowiska

    4. Projekty i inicjatywy:
       - Dodaj sekcję pokazującą praktyczne zastosowanie umiejętności
       - Uwzględnij projekty osobiste, wolontariat, działalność dodatkową
       - Podkreśl rezultaty i zdobyte kompetencje

    CV:
    {cv_text}

    OPIS STANOWISKA:
    {job_description}

    WAŻNE:
    1. Zachowaj pełną prawdziwość informacji
    2. Nie wymyślaj doświadczenia ani umiejętności
    3. Skup się na transformacji i lepszym przedstawieniu istniejących kompetencji
    4. Używaj języka zrozumiałego dla branży
    5. Podkreśl potencjał i chęć rozwoju

    Zwróć zoptymalizowane CV w formacie tekstowym, wykorzystując powyższe wytyczne do maksymalnego wydobycia wartości z istniejącego doświadczenia kandydata.
    """

    return send_api_request(prompt, max_tokens=3000)
    # Jeśli nie podano słów kluczowych, spróbuj je wygenerować
    if keywords_data is None:
        try:
            keywords_data = extract_keywords_from_job(job_description)
        except Exception as e:
            logger.error(f"Failed to extract keywords for CV optimization: {str(e)}")
            keywords_data = {}

    # Wykryj poziom doświadczenia, branżę, typ pracy i konkretną rolę
    try:
        seniority = detect_seniority_level(cv_text, job_description)
        logger.info(f"Detected seniority level: {seniority}")

        industry = detect_industry(job_description)
        logger.info(f"Detected industry: {industry}")

        job_type = detect_job_type(job_description)
        logger.info(f"Detected job type: {job_type}")

        specific_role = detect_specific_role(job_description)
        logger.info(f"Detected specific role: {specific_role}")
    except Exception as e:
        logger.error(f"Error detecting context: {str(e)}")
        seniority = "mid"  # Domyślny poziom
        industry = "general"  # Domyślna branża
        job_type = "office"  # Domyślny typ pracy
        specific_role = "specjalista"  # Domyślna rola

    # Pobierz specyficzne wytyczne dla branży, poziomu, typu pracy i roli
    industry_prompt = get_industry_specific_prompt(industry, seniority, job_type, specific_role)
    achievements_prompt = get_measurable_achievements_prompt(seniority)
    structural_prompt = get_structural_quality_control_prompt(seniority, industry)

    # Przygotuj dodatkowe wytyczne na podstawie słów kluczowych
    keyword_instructions = ""

    if keywords_data and isinstance(keywords_data, dict):
        keyword_instructions = "KLUCZOWE SŁOWA, KTÓRE NALEŻY UWZGLĘDNIĆ:\n\n"

        # Dodaj wysokopriorytetowe słowa kluczowe
        high_priority_keywords = []

        for category, words in keywords_data.items():
            category_name = category.replace("_", " ").title()
            for word in words:
                if isinstance(word, dict) and "slowo" in word and "waga" in word:
                    if word["waga"] >= 4:  # Wysoki priorytet
                        high_priority_keywords.append(f"{word['slowo']} ({category_name})")

        if high_priority_keywords:
            keyword_instructions += "Najważniejsze słowa kluczowe (koniecznie uwzględnij):\n"
            for kw in high_priority_keywords:
                keyword_instructions += f"- {kw}\n"
            keyword_instructions += "\n"

        # Dodaj kategoryzowane słowa
        for category, words in keywords_data.items():
            if words:
                category_name = category.replace("_", " ").title()
                keyword_instructions += f"{category_name}:\n"

                for word in words:
                    if isinstance(word, dict) and "slowo" in word:
                        keyword_instructions += f"- {word['slowo']}\n"

                keyword_instructions += "\n"

    # Pobierz szczegółowe informacje o kompetencjach dla danej roli
    role_competencies = get_role_specific_competencies(specific_role)

    # Przygotuj sugestie dotyczące brakujących kompetencji
    missing_competencies_suggestions = """
    SUGESTIE DOTYCZĄCE POTENCJALNIE BRAKUJĄCYCH KOMPETENCJI:
    Jeśli poniższe kompetencje nie występują w oryginalnym CV, a są istotne dla danej roli, umieść dodatkową sekcję
    z sugestiami, które kandydat mógłby dodać jeśli je posiada:
    """

    for cert in role_competencies.get("certifications", [])[:3]:  # Wybierz maksymalnie 3 najważniejsze
        missing_competencies_suggestions += f"- {cert}\n"

    for skill in role_competencies.get("skills", [])[:3]:  # Wybierz maksymalnie 3 najważniejsze
        missing_competencies_suggestions += f"- {skill}\n"

    prompt = f"""
    TASK: Stwórz całkowicie nową, profesjonalną wersję CV, które wyróżni kandydata na tle konkurencji. CV musi być precyzyjnie dopasowane do wymagań stanowiska, zawierać naturalne, poprawne językowo sformułowania i podkreślać najważniejsze osiągnięcia i umiejętności.

    UWAGA - KRYTYCZNE WYMAGANIA JĘZYKOWE:
    1. Używaj WYŁĄCZNIE naturalnego, codziennego języka polskiego
    2. UNIKAJ wymyślonych słów i dziwacznych sformułowań
    3. NIE TWÓRZ neologizmów ani nietypowych połączeń wyrazowych
    4. UNIKAJ bardzo formalnych, akademickich zwrotów
    5. Sprawdź czy każde zdanie brzmi NATURALNIE po polsku
    6. NIE UŻYWAJ słów, których nie znalazłbyś w słowniku języka polskiego

    Wykryty poziom doświadczenia: {seniority.upper()}
    Wykryta branża: {industry.upper()}
    Wykryty typ pracy: {job_type.upper()}
    Wykryta konkretna rola: {specific_role.upper()}

    WSKAZÓWKI PROFESJONALNEGO FORMATOWANIA:

    1. Rozpocznij od mocnego, ukierunkowanego na stanowisko podsumowania zawodowego (3-4 zdania), które:
       - Natychmiast przyciągnie uwagę rekrutera
       - Podkreśli najważniejsze kwalifikacje odpowiadające stanowisku
       - Zawiera 2-3 najważniejsze osiągnięcia z liczbami/procentami
       - Jest napisane w pierwszej osobie, aktywnym językiem

    2. FORMATOWANIE SEKCJI UMIEJĘTNOŚCI - zastosuj nowoczesne podejście:

       a) Umiejętności twarde (techniczne/specjalistyczne) z oznaczeniem "**" i poziomem zaawansowania:
          **Umiejętności techniczne:**
          - [Kluczowa umiejętność 1 powiązana ze stanowiskiem] (Zaawansowany)
          - [Kluczowa umiejętność 2 powiązana ze stanowiskiem] (Średniozaawansowany)
          - [Umiejętność techniczna 3] (Podstawowy)

       b) Umiejętności miękkie z oznaczeniem "**" i konkretnym zastosowaniem:
          **Umiejętności interpersonalne:**
          - [Umiejętność miękka 1] (z przykładem zastosowania)
          - [Umiejętność miękka 2] (z przykładem zastosowania)

       c) Umiejętności branżowe z oznaczeniem "**" - specyficzne dla danej branży:
          **Umiejętności branżowe:**
          - [Specjalistyczna umiejętność branżowa 1]
          - [Specjalistyczna umiejętność branżowa 2]

    3. Dodatkowe umiejętności również wydziel w osobnej sekcji z pogrubieniem, ale bardziej skonkretyzowane:
       **Dodatkowe umiejętności:**
       - [Konkretna dodatkowa umiejętność 1 wspierająca główne kwalifikacje]
       - [Konkretna dodatkowa umiejętność 2 zwiększająca konkurencyjność kandydata]

    4. Certyfikaty i kwalifikacje umieść w osobnej sekcji z pogrubieniem i datami uzyskania:
       **CERTYFIKATY I KWALIFIKACJE:**
       - [Nazwa certyfikatu 1] (Rok uzyskania)
       - [Nazwa certyfikatu 2] (Rok uzyskania - ważny do [Rok])

    5. STRUKTURA CAŁEGO CV - zapewnij ujednolicony format:
       - Zastosuj spójny układ graficzny
       - Używaj jednolitych nagłówków sekcji
       - Zadbaj o czytelne odstępy między sekcjami
       - Usuń błędy ortograficzne i interpunkcyjne
       - Sprawdź poprawność językową
       - Usuń całkowicie niepotrzebne informacje

    6. ANALIZA BRAKUJĄCYCH ELEMENTÓW:
       1. Przeanalizuj wymagania ze stanowiska i porównaj z obecnym CV
       2. Zidentyfikuj brakujące umiejętności techniczne i miękkie
       3. Zaproponuj dodatkowe kwalifikacje i certyfikaty
       4. Sugeruj konkretne kursy i szkolenia do uzupełnienia
       5. Wskaż obszary doświadczenia do rozwinięcia

    Format CV musi być PROFESJONALNY I CZYTELNY:

    (Na początku CV)
    # IMIĘ NAZWISKO
    **Stanowisko, o które się ubiegam**

    **Kontakt:** telefon | email | miejscowość

    **Podsumowanie zawodowe:**
    Zwięzły, profesjonalny opis zawierający najważniejsze kwalifikacje i osiągnięcia (3-4 zdania).

    **DOŚWIADCZENIE ZAWODOWE:**

    **Nazwa stanowiska** | Firma, Lokalizacja | Okres zatrudnienia
    Krótki opis firmy (1 zdanie).
    • Osiągnięcie 1 z mierzalnym rezultatem
    • Osiągnięcie 2 z mierzalnym rezultatem
    • Osiągnięcie 3 z mierzalnym rezultatem

    **UMIEJĘTNOŚCI:**

    **Umiejętności techniczne:**
    • Umiejętność 1 (poziom zaawansowania)
    • Umiejętność 2 (poziom zaawansowania)

    **Umiejętności interpersonalne:**
    • Umiejętność 1
    • Umiejętność 2

    **WYKSZTAŁCENIE:**

    **Nazwa uczelni/szkoły** | Kierunek/specjalizacja | Okres

    **CERTYFIKATY I KWALIFIKACJE:**
    • Certyfikat 1 (rok uzyskania)
    • Certyfikat 2 (rok uzyskania)

    **SUGESTIE ROZWOJU:**

    **Umiejętności do zdobycia:**
    • Umiejętność 1 (wysoki priorytet)
    • Umiejętność 2 (średni priorytet)

    **Rekomendowane certyfikaty:**
    • Certyfikat 1
    • Certyfikat 2

    SZCZEGÓŁOWE WYTYCZNE DLA SEKCJI DOŚWIADCZENIA ZAWODOWEGO:

    SEKCJA DOŚWIADCZENIA ZAWODOWEGO - ZASADY MISTRZOWSKIEGO CV:

    UWAGA: KONIECZNIE PRZEANALIZUJ I KOMPLEKSOWO PRZEOBRAŹ OPISY DOŚWIADCZENIA Z ORYGINALNEGO CV!
    Nie kopiuj, ale całkowicie przetwórz treść, zachowując prawdziwe fakty i chronologię.

    1. Mistrzowskie opisy stanowisk:
       - ROZPOCZNIJ każdy punkt MOCNYM CZASOWNIKIEM EFEKTU (zoptymalizowałem, zwiększyłem, wdrożyłem)
       - OBOWIĄZKOWO dołącz LICZBY i PRECYZYJNE DANE (zwiększenie X o Y%, obsługa Z klientów miesięcznie)
       - ZASTOSUJ metodę STAR (Sytuacja, Zadanie, Działanie, Rezultat) dla najważniejszych osiągnięć
       - PODKREŚL wymierne rezultaty, które przyniosły KORZYŚCI dla pracodawcy (oszczędności, wzrost efektywności)
       - DOPASUJ każde doświadczenie do wymagań NOWEGO stanowiska

    2. Profesjonalna struktura każdego stanowiska:
       - **Nazwa stanowiska:** [Precyzyjna, marketingowa i zgodna ze standardami branżowymi]
       - **Firma | Lokalizacja:** [Nazwa firmy + miasto lub region]
       - **Okres zatrudnienia:** [MM.RRRR - MM.RRRR] lub [MM.RRRR - obecnie]
       - **Krótki opis firmy:** [1 zdanie o profilu działalności, wielkości, zasięgu]
       - **Najważniejsze osiągnięcia i odpowiedzialności:**
         • [Mocny czasownik + konkretne zadanie + mierzalny rezultat + korzyść dla firmy]
         • [Mocny czasownik + konkretne zadanie + mierzalny rezultat + korzyść dla firmy]
         • [Mocny czasownik + konkretne zadanie + mierzalny rezultat + korzyść dla firmy]

    3. Strategiczna transformacja treści:
       - CAŁKOWICIE PRZEFORMUŁUJ każdy element doświadczenia pod kątem NOWEGO stanowiska
       - UŻYWAJ wyłącznie AKTYWNYCH konstrukcji językowych (nigdy "byłem odpowiedzialny za")
       - WŁĄCZ słowa kluczowe i terminologię z OGŁOSZENIA o pracę (minimum 5-7 kluczowych terminów)
       - STOSUJ JĘZYK BRANŻOWY właściwy dla poziomu stanowiska
       - OGRANICZ każdy punkt do maksymalnie 2 linijek tekstu
       - USUŃ wszystko, co nie wspiera Twojej kandydatury na NOWE stanowisko

    4. Przykłady transformacji - najwyższy poziom profesjonalizmu:

       PRZED: "Odpowiadałem za obsługę klienta i realizację zamówień."
       PO: "Obsłużyłem średnio 45 klientów dziennie (największa liczba w zespole), utrzymując 98% wskaźnik satysfakcji i skracając średni czas realizacji zamówień o 15%, co przełożyło się na wzrost sprzedaży o 8% w ciągu kwartału."

       PRZED: "Kierowałem samochodem dostawczym."
       PO: "Zoptymalizowałem 15 tras dostaw w aglomeracji warszawskiej, co skróciło czas dostawy o 12% i zmniejszyło zużycie paliwa o 8%, realizując 25-30 punktów dziennie z bezbłędną dokumentacją i 100% terminowością przez 24 miesiące."

       PRZED: "Zajmowałem się organizacją magazynu."
       PO: "Przeprojektowałem układ magazynu o powierzchni 1200m² wraz z systemem oznakowania i strefowania, co zwiększyło przepustowość o 27%, skróciło czas kompletacji o 22% i zredukowało liczbę błędów o 35%, przynosząc oszczędności rzędu 15 000 zł miesięcznie."

    Kluczowe wytyczne optymalizacji:
    1. Głęboka analiza i transformacja doświadczenia:
       - OBOWIĄZKOWO przeprowadź szczegółową analizę każdego stanowiska z CV pod kątem wymagań nowej roli
       - KONIECZNIE zidentyfikuj i wyeksponuj transferowalne umiejętności
       - ZAWSZE dodawaj wymierzalne rezultaty i osiągnięcia (%, liczby, skala projektów)
       - STOSUJ profesjonalne, branżowe słownictwo charakterystyczne dla danego sektora
       - TWÓRZ całkowicie nowe opisy stanowisk wykorzystując słowa kluczowe z ogłoszenia

    2. Zaawansowana personalizacja:
       - Dopasuj tone of voice do kultury firmy i branży
       - Uwzględnij specyficzne technologie i metodologie wymienione w ogłoszeniu
       - Dodaj sekcję highlight'ów dopasowaną do priorytetowych wymagań
       - Stwórz spersonalizowane podsumowanie zawodowe podkreślające najważniejsze atuty

    3. Optymalizacja umiejętności i kompetencji:
       - Podziel umiejętności na kategorie: techniczne, miękkie, branżowe
       - Określ poziom zaawansowania w skali 1-5 dla kluczowych kompetencji
       - Dodaj konkretne przykłady zastosowania każdej kluczowej umiejętności
       - Uwzględnij certyfikaty i szkolenia istotne dla stanowiska

    4. Spójność i logika danych:
       - Sprawdź, czy daty są logiczne i zachowują ciągłość
       - Upewnij się, że ścieżka kariery jest spójna (brak nielogicznych przeskoków)
       - Zadbaj o realistyczne opisy osiągnięć (liczby, procenty)
       - Dopasuj poziom stanowisk do wykrytego seniority

    5. Wytyczne branżowo-specyficzne i dotyczące roli:
    {industry_prompt}

    6. Wytyczne odnośnie mierzalnych osiągnięć:
    {achievements_prompt}

    7. Wytyczne dotyczące struktury i jakości:
    {structural_prompt}

    8. {missing_competencies_suggestions}

    {keyword_instructions}

    WAŻNE ZASADY:
    - Zachowaj pełną spójność z prawdą zawartą w oryginalnym CV
    - Każda sekcja musi być napisana od nowa z fokusem na nowe stanowisko
    - Używaj aktywnych czasowników i konkretnych przykładów
    - Odpowiedz w tym samym języku co oryginalne CV
    - KONIECZNIE uwzględnij najważniejsze słowa kluczowe wymienione powyżej
    - Stwórz CV w formacie odpowiednim dla wykrytego typu pracy i branży

    DANE:

    Opis stanowiska:
    {job_description}

    Oryginalne CV:
    {cv_text}

    Zwróć tylko zoptymalizowane CV w formacie tekstowym, bez dodatkowych komentarzy.
    """

    return send_api_request(prompt, max_tokens=2500)

def optimize_cv(cv_text, job_description):
    """
    Create an optimized version of CV using advanced AI processing
    """
    # Dla zachowania kompatybilności, wywołaj nową funkcję
    return optimize_cv_with_keywords(cv_text, job_description)

def generate_recruiter_feedback(cv_text, job_description=""):
    """
    Generate feedback on a CV as if from an AI recruiter
    """
    context = ""
    if job_description:
        context = f"Job description for context:\n{job_description}"

    # Sprawdź czy CV jest w języku polskim
    is_polish = len([word for word in ["jestem", "doświadczenie", "umiejętności", "wykształcenie", "praca", "stanowisko", "firma", "uniwersytet", "szkoła", "oraz", "język", "polski"] if word.lower() in cv_text.lower()]) > 3

    prompt = f"""
    TASK: Jesteś doświadczonym rekruterem. Przeanalizuj to CV i dostarcz szczegółowej, praktycznej informacji zwrotnej.

    {"UWAGA: TO CV JEST W JĘZYKU POLSKIM. ODPOWIEDZ KONIECZNIE PO POLSKU!" if is_polish else "This CV appears to be in English. Please respond in English."}

    Uwzględnij:
    1. Ogólne wrażenie
    2. Mocne i słabe strony
    3. Ocena formatowania i struktury
    4. Ocena jakości treści
    5. Kompatybilność z systemami ATS
    6. Konkretne sugestie ulepszeń
    7. Ocena w skali 1-10

    BARDZO WAŻNE: Odpowiedz w tym samym języku co CV. Jeśli CV jest po polsku, odpowiedz po polsku. Jeśli CV jest po angielsku, odpowiedz po angielsku.

    {context}

    CV:
    {cv_text}

    Dostarcz szczegółowej opinii rekrutera. Bądź szczery, ale konstruktywny.
    """

    return send_api_request(prompt, max_tokens=2000)

def generate_cover_letter(cv_text, job_description, company_name="", company_culture=""):
    """
    Generate an advanced, personalized cover letter with company research and cultural fit
    """
    prompt = f"""
    TASK: Stwórz spersonalizowany list motywacyjny uwzględniający:

    1. Specyfikę firmy i jej kulturę organizacyjną
    2. Dopasowanie doświadczenia do wartości firmy
    3. Elementy storytellingu pokazujące motywację
    4. Konkretne przykłady osiągnięć powiązane ze stanowiskiem
    5. Odniesienia do aktualnych projektów/wyzwań firmy

    Firma: {company_name}
    Kultura organizacyjna: {company_culture}
    """
    prompt = f"""
    TASK: Create a personalized cover letter based on this CV and job description.

    The cover letter should:
    - Be professionally formatted
    - Highlight relevant skills and experiences from the CV
    - Connect the candidate's background to the job requirements
    - Include a compelling introduction and conclusion
    - Be approximately 300-400 words

    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.

    Job description:
    {job_description}

    CV:
    {cv_text}

    Return only the cover letter in plain text format.
    """

    return send_api_request(prompt, max_tokens=2000)

def translate_to_english(cv_text):
    """
    Translate a CV to English while preserving professional terminology
    """
    prompt = f"""
    TASK: Translate this CV to professional English.

    Important:
    - Maintain all professional terminology
    - Preserve the original structure and formatting
    - Ensure proper translation of industry-specific terms
    - Keep names of companies and products unchanged
    - Make sure the translation sounds natural and professional in English

    Original CV:
    {cv_text}

    Return only the translated CV in plain text format.
    """

    return send_api_request(prompt, max_tokens=2500)

def suggest_alternative_careers(cv_text):
    """
    Suggest alternative career paths based on the skills in a CV
    """
    prompt = f"""
    TASK: Analyze this CV and suggest alternative career paths based on the skills and experience.

    For each suggested career path include:
    1. Job title/role
    2. Why it's a good fit based on existing skills
    3. What additional skills might be needed
    4. Potential industries or companies to target
    5. Estimated effort to transition (low/medium/high)

    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.

    CV:
    {cv_text}

    Provide a detailed analysis with specific, actionable recommendations.
    """

    return send_api_request(prompt, max_tokens=2000)

def generate_multi_versions(cv_text, roles):
    """
    Generate multiple versions of a CV, each precisely tailored for specific roles
    with advanced formatting and role-specific emphasis
    """
    roles_text = "\n".join([f"- {role}" for role in roles])

    prompt = f"""
    TASK: Stwórz profesjonalnie dostosowane wersje CV dla każdej roli.

    Role do przygotowania:
    {roles_text}

    Dla każdej wersji CV wykonaj:

    1. ANALIZA ROLI:
       - Zidentyfikuj kluczowe wymagania dla danej roli
       - Określ najważniejsze kompetencje i umiejętności
       - Zbadaj specyfikę branży i jej oczekiwania

    2. TRANSFORMACJA DOŚWIADCZENIA:
       - Przekształć każde doświadczenie pod kątem danej roli
       - Wydobądź i podkreśl relevant achievements
       - Dostosuj język do standardów branżowych
       - Usuń lub zmniejsz nacisk na nieistotne elementy

    3. DOSTOSOWANIE STRUKTURY:
       - Zmień kolejność sekcji pod kątem priorytetów roli
       - Dodaj sekcje specyficzne dla danego stanowiska
       - Dostosuj format do standardów branżowych
       - Zoptymalizuj pod kątem ATS dla danej branży

    4. WZMOCNIENIE KLUCZOWYCH ELEMENTÓW:
       - Dodaj sekcję highlight'ów specyficznych dla roli
       - Podkreśl certyfikaty i szkolenia istotne dla stanowiska
       - Zaakcentuj projekty związane z daną rolą
       - Uwypuklij osiągnięcia najbardziej relevant dla pozycji

    5. PROFESJONALNA PERSONALIZACJA:
       - Dostosuj ton i styl języka do kultury branżowej
       - Użyj terminologii specyficznej dla danej roli
       - Dodaj branżowe słowa kluczowe
       - Zachowaj spójność z wymaganiami rynku

    ZASADY FORMATOWANIA:
    - Użyj czytelnego, profesjonalnego układu
    - Zachowaj spójny system nagłówków
    - Zastosuj odpowiednie odstępy i marginesy
    - Zadbaj o hierarchię informacji

    WAŻNE WSKAZÓWKI:
    - Zachowaj pełną prawdziwość informacji
    - Nie wymyślaj doświadczenia ani umiejętności
    - Skup się na transformacji i lepszym przedstawieniu istniejących kompetencji
    - Każda wersja powinna być kompletna i samodzielna

    Oryginalne CV:
    {cv_text}

    Zwróć każdą wersję CV oddzieloną wyraźnym nagłówkiem z nazwą roli, zachowując pełen profesjonalizm i spójność.
    """

    return send_api_request(prompt, max_tokens=3500)

def analyze_job_url(url):
    """
    Extract job description from a URL with improved handling forjson" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("