"""
lyrics_search.py - Online Lyrics Search & Confirmation Module
Automatically looks up lyrics on LyricsTranslate.com and search engines,
extracting original text, community transliterations, and translations.
"""

import os
import re
import urllib.parse
import urllib.request
import json
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def clean_song_query(title: str, artist: str = "") -> str:
    """Remove boilerplate video words, album suffixes (// ...), Vol. tags, and years."""
    # Keep primary song segment before '//' or '|' metadata blocks
    cleaned = re.split(r"\s*(?://|\|)\s*", title)[0]
    patterns = [
        r"\(Official.*?\)", r"\[Official.*?\]",
        r"\(Audio.*?\)", r"\[Audio.*?\]",
        r"\(Video.*?\)", r"\[Video.*?\]",
        r"\(Lyric.*?\)", r"\[Lyric.*?\]",
        r"\(Clip.*?\)", r"\[Clip.*?\]",
        r"\bVol\.?\s*\d+\b",
        r"\b(?:19|20)\d{2}\b",
        r"\b(?:HD|4K|1080p|HQ)\b"
    ]
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\-_|/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if artist and artist != "Unknown Artist" and artist.lower() not in cleaned.lower():
        return f"{artist} {cleaned}".strip()
    return cleaned

def fetch_html(url: str) -> Optional[str]:
    """Fetch HTML content with standard browser user-agent and SSL context."""
    try:
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_ctx = ssl._create_unverified_context()

        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        )
        with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return None

def search_lyricstranslate(query: str) -> List[Dict[str, Any]]:
    """
    Search LyricsTranslate for matching songs.
    Returns list of candidate matches with title, artist, url.
    """
    candidates = []
    # LyricsTranslate search query endpoint
    search_url = f"https://lyricstranslate.com/en/site-search?query={urllib.parse.quote(query)}"
    html = fetch_html(search_url)
    if not html:
        return candidates

    soup = BeautifulSoup(html, "html.parser")
    # Parse search result links
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)
        # Song pages on lyricstranslate typically look like /en/<artist>-<song>-lyrics.html
        if "-lyrics.html" in href and text and len(text) > 3:
            full_url = href if href.startswith("http") else f"https://lyricstranslate.com{href}"
            if not any(c["url"] == full_url for c in candidates):
                candidates.append({
                    "title": text,
                    "url": full_url,
                    "source": "LyricsTranslate"
                })
        if len(candidates) >= 5:
            break

    return candidates

def extract_lyricstranslate_song(url: str) -> Dict[str, Any]:
    """
    Extract lyrics text, transliteration (if present), and translations from a LyricsTranslate page.
    """
    res = {
        "url": url,
        "original_lyrics": "",
        "transliterated_lyrics": "",
        "english_translation": "",
        "found": False
    }
    html = fetch_html(url)
    if not html:
        return res

    soup = BeautifulSoup(html, "html.parser")
    
    # Original lyrics container
    song_body = soup.find("div", class_=re.compile(r"song-node-text|parole"))
    if song_body:
        # Replace <br> with newlines
        for br in song_body.find_all("br"):
            br.replace_with("\n")
        res["original_lyrics"] = song_body.get_text().strip()
        res["found"] = bool(res["original_lyrics"])

    # Look for existing transliterations or English translations linked on the page
    for trans_link in soup.find_all("a", href=True):
        href = trans_link["href"]
        text = trans_link.get_text(strip=True).lower()
        if "transliteration" in text and not res["transliterated_lyrics"]:
            trans_html = fetch_html(href if href.startswith("http") else f"https://lyricstranslate.com{href}")
            if trans_html:
                tsoup = BeautifulSoup(trans_html, "html.parser")
                tbody = tsoup.find("div", class_=re.compile(r"song-node-text|translate-node-text"))
                if tbody:
                    for br in tbody.find_all("br"):
                        br.replace_with("\n")
                    res["transliterated_lyrics"] = tbody.get_text().strip()

    return res

def auto_search_lyrics(title: str, artist: str = "") -> Dict[str, Any]:
    """
    Main entry point: cleans query and checks verified LyricsTranslate pages.
    Does NOT guess lyrics from title text alone (prevents LLM hallucinations);
    when not found, the pipeline transcribes & times directly from the vocal audio.
    """
    query = clean_song_query(title, artist)
    candidates = search_lyricstranslate(query)
    
    best_match = {
        "query": query,
        "found": False,
        "original_lyrics": "",
        "source_url": "",
        "candidates": candidates
    }
    
    if candidates:
        top_url = candidates[0]["url"]
        extracted = extract_lyricstranslate_song(top_url)
        if extracted.get("found"):
            best_match["found"] = True
            best_match["original_lyrics"] = extracted["original_lyrics"]
            best_match["transliterated_lyrics"] = extracted.get("transliterated_lyrics", "")
            best_match["source_url"] = top_url
            return best_match
            
    return best_match

def _read_multiline_paste() -> str:
    print("\nEnter/paste official lyrics below (press Enter on empty line or Ctrl+D to finish):")
    lines = []
    while True:
        try:
            line = input()
            if not line and lines:
                break
            if line:
                lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def confirm_lyrics_interactive(lyrics_result: Dict[str, Any]) -> str:
    """
    Interactive CLI helper to allow the user to confirm online reference lyrics,
    transcribe directly from the singer's vocal audio, or paste manual lyrics.
    """
    if lyrics_result.get("found"):
        print("\n" + "="*60)
        print(" [CANDIDATE LYRICS FOUND ONLINE]")
        print(f" Source: {lyrics_result.get('source_url')}")
        print(" Preview (first 6 lines):")
        preview = "\n".join(lyrics_result["original_lyrics"].split("\n")[:6])
        print(preview)
        print("="*60)
        choice = input("\nUse as spelling reference while timing on vocals? [Y / a=audio only / p=paste manual]: ").strip()
        if choice.lower() in ("", "y", "yes"):
            return lyrics_result["original_lyrics"]
        elif choice.lower() in ("paste", "manual", "p"):
            return _read_multiline_paste()
        else:
            return ""
    else:
        print("\n" + "="*60)
        print(f" [NO VERIFIED LYRICS ONLINE FOR: '{lyrics_result.get('query', '')}']")
        print(" Will listen directly to the singer's vocal track to extract & time the exact lyrics.")
        print("="*60)
        choice = input("\nPress Enter to auto-transcribe from vocals, or type 'p' to paste lyrics manually [Enter/p]: ").strip()
        if choice.lower() in ("paste", "manual", "p", "y", "yes"):
            return _read_multiline_paste()

    return ""

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Amr Diab Nour El Ein"
    print(f"Searching lyrics for: {q}")
    res = auto_search_lyrics(q)
    print("Found:", res["found"], "URL:", res.get("source_url"))
    if res["found"]:
        print(res["original_lyrics"][:200] + "...")
