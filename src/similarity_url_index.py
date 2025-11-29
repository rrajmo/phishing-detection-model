from typing import Tuple, List, Optional
import pandas as pd
from pathlib import Path
import tldextract

def load_domains_from_csv(file: str, nrows: int = 1000) -> List[str]:
    df = pd.read_csv(file, usecols=["Domain"], nrows=nrows)
    domains = df["Domain"].dropna().astype(str).str.lower().tolist()
    return domains

BASE_DIR = Path(__file__).resolve().parents[1]
domains = load_domains_from_csv(BASE_DIR / "domains/top1thousanddomains.csv", 1000)
                                
def order_urls_by_length(source: str, target: str) -> Tuple[str, str, int]:
    if len(source) < len(target):
        return source, target, len(source)
    else:
        return target, source, len(target)
    
def calculate_similarity_url_index(source: str, target: str) -> float:
    if not source or not target:
        return 0
    
    short_url, long_url, length_of_short_url = order_urls_by_length(source, target)
    length_of_long_url = max(len(source), len(target))
    similarity_index = 0
    base_value = 50 / length_of_long_url
    sum_of_natural_numbers = (length_of_long_url * (length_of_long_url + 1)) / 2
    i = 0
    while i < length_of_short_url:
        if short_url[i] == long_url[i]:
            weight = (50 * (length_of_long_url - i)) / sum_of_natural_numbers
            similarity_index += base_value + weight
        else:
            long_url = long_url[:i] + long_url[i + 1:]
            short_url, long_url, length_of_short_url = order_urls_by_length(short_url, long_url)
            i -= 1
        i += 1
    return round(similarity_index, 2)

def extract_domain(url: str) -> Optional[str]:
    try:
        parts = tldextract.extract(url)
        if not parts.domain:
            return None
        domain = f"{parts.domain}".lower()
        return domain
    except Exception:
        return None
    
def calculate_maximum_similarity_url_index(url: str, domains: List[str]) -> float:
    url_domain = extract_domain(url)
    if not url_domain:
        return 0
    
    maximum_similarity_url_index = 0
    for domain in domains:
        if url_domain == domain:
            return 100
        similarity_url_index = calculate_similarity_url_index(url_domain, domain)
        maximum_similarity_url_index = max(maximum_similarity_url_index, similarity_url_index)

    return maximum_similarity_url_index

def get_maximum_similarity_url_index(url: str) -> float:
    return calculate_maximum_similarity_url_index(url, domains)