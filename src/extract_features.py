from urllib.parse import urlparse
import tldextract
import ipaddress
import re

def extract_url_length(url: str) -> int:
    try:
        return len(url)
    except Exception:
        return 0

def extract_domain_length(url: str) -> int:
    try: 
        parsed_url = tldextract.extract(url)
        domain = parsed_url.domain
        return len(domain) if domain else 0
    except Exception:
        return 0

def is_domain_ip_address(url: str) -> int:
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if not hostname:
            return 0
        
        try:
            ipaddress.ip_address(hostname)
            return 1
        except Exception:
            return 0
    except Exception:
        return 0
    
def extract_character_continuation_rate(url: str) -> float:
    total_length = extract_url_length(url)
    if total_length == 0:
        return 0

    alphabet_sequences = re.findall(r'[a-zA-Z]+', url)
    longest_alphabet_sequence = max((len(sequence) for sequence in alphabet_sequences), default=0)

    digit_sequences = re.findall(r'\d+', url)
    longest_digit_sequence = max((len(sequence) for sequence in digit_sequences), default=0)

    special_sequences = re.findall(r'[^a-zA-Z0-9]+', url)
    longest_special_sequence = max((len(sequence) for sequence in special_sequences), default=0)

    continuation_sum = longest_alphabet_sequence + longest_digit_sequence + longest_special_sequence
    return round(continuation_sum / total_length, 4)

def extract_tld_length(url: str) -> int:
    try: 
        url_components = tldextract.extract(url)
        suffix = url_components.suffix
        if not suffix:
            return 0
        return len(suffix)
    except Exception:
        return 0

def extract_number_of_subdomains(url: str) -> int:
    try:
        url_components = tldextract.extract(url)
        subdomain = url_components.subdomain
        if not subdomain:
            return 0
        return len(subdomain.split("."))
    except:
        return 0

def extract_has_obfuscation(url: str) -> int:
    return 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0

def extract_number_of_obfuscated_characters(url: str) -> int:
    obfuscated_matches = re.findall(r'%[0-9a-fA-F]{2}', url)
    return len(obfuscated_matches)

def extract_obfuscation_ratio(url: str) -> int:
    number_of_obfuscated_matches = extract_number_of_obfuscated_characters(url)
    total_length = extract_url_length(url)
    if total_length == 0:
        return 0
    return round(number_of_obfuscated_matches / total_length, 4)

def extract_number_of_letters(url: str) -> int:
    try:
        url_components = tldextract.extract(url)
        subdomain = ""
        if '.' in url_components.subdomain:
            subdomain = url_components.subdomain.split(".", 1)[-1]
        components = [subdomain, url_components.domain or "", url_components.suffix or ""]
        total = 0
        for component in components:
            for character in component:
                if character.isalpha():
                    total += 1
        return total
    except Exception:
        return 0

def extract_letter_ratio(url: str) -> float:
    try:
        url_length = extract_url_length(url)
        number_of_letters = extract_number_of_letters(url)
        if url_length == 0:
            return 0
        return round(number_of_letters / url_length, 4)
    except Exception:
        return 0.0

def extract_number_of_digits(url: str) -> int:
    try:
        url_components = tldextract.extract(url)
        subdomain = ""
        if '.' in url_components.subdomain:
            subdomain = url_components.subdomain.split(".", 1)[-1]
        components = [subdomain, url_components.domain or "", url_components.suffix or ""]
        total = 0
        for component in components:
            for digit in component:
                if digit.isdigit():
                    total += 1
        return total
    except Exception:
        return 0

def extract_digit_ratio(url: str) -> float:
    try:
        url_length = extract_url_length(url)
        number_of_digits = extract_number_of_digits(url)
        if url_length == 0:
            return 0
        return round(number_of_digits / url_length, 4)
    except Exception:
        return 0.0

def extract_number_of_equals_sign(url: str) -> int:
    return url.count("=")

def extract_number_of_question_mark(url: str) -> int:
    return url.count("?")

def extract_number_of_ampersand(url: str) -> int:
    return url.count("&")

def is_https(url: str) -> int:
    try:
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme or ""
        return 1 if scheme.lower() == "https" else 0
    except Exception:
        return 0